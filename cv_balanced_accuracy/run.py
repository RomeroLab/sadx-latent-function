#!/usr/bin/env python3
"""Run grouped CV ablations and print only their balanced accuracy table."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr
import hashlib
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODELS = ['full', 'no_dca', 'no_library_bias', 'sequence_only', 'dca_only',
          'library_prior', 'always_low']
LABELS = ['Full model', 'No DCA', 'No library bias', 'Sequence only', 'DCA only',
          'Library-prior baseline', 'Always-Low baseline']


def resolve(path):
    """All relative CLI paths are relative to the repository, never the cwd."""
    return path if path.is_absolute() else ROOT / path


def table(predictions):
    if 'evaluation' in predictions:
        predictions = predictions[predictions.evaluation.eq('cross_validation')].copy()
    required = {'model', 'seed', 'row_index', 'parent', 'y_true',
                'prob_N', 'prob_L', 'prob_P', 'prob_H'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    if set(predictions.model) != set(MODELS):
        raise ValueError('Expected all five ablations and both baselines.')
    if predictions.duplicated(['model', 'seed', 'row_index']).any():
        raise ValueError('Expected one out-of-fold prediction per row, model, and seed.')
    probs = predictions[['prob_N', 'prob_L', 'prob_P', 'prob_H']].to_numpy()
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1, atol=2e-6):
        raise ValueError('Invalid class probability vectors.')
    reference = predictions[predictions.model.eq('full')]
    seeds = set(reference.seed)
    expected = set(reference.row_index)
    truth = reference.drop_duplicates('row_index').set_index('row_index')[['parent', 'y_true']].sort_index()
    for name in MODELS:
        if set(predictions.loc[predictions.model.eq(name), 'seed']) != seeds:
            raise ValueError(f'Incomplete seed coverage for {name}.')
    results = []
    for (name, seed), group in predictions.groupby(['model', 'seed']):
        if set(group.row_index) != expected:
            raise ValueError(f'Incomplete sequence coverage for {name}, seed {seed}.')
        pd.testing.assert_frame_equal(group.set_index('row_index')[['parent', 'y_true']].sort_index(), truth)
        for parent in ['1VH', '2L', '3VRL', 'All']:
            part = group if parent == 'All' else group[group.parent.eq(parent)]
            if set(part.y_true) != {0, 1, 2, 3}:
                raise ValueError(f'{parent} must contain all four classes for this table.')
            predicted = part[['prob_N', 'prob_L', 'prob_P', 'prob_H']].to_numpy().argmax(axis=1)
            score = recall_score(part.y_true, predicted, labels=[0, 1, 2, 3], average='macro', zero_division=0)
            results.append((name, seed, parent, score))
    frame = pd.DataFrame(results, columns=['model', 'seed', 'parent', 'score'])
    scores = frame.groupby(['model', 'parent']).score.mean().unstack().reindex(index=MODELS, columns=['1VH', '2L', '3VRL', 'All'])
    scores.index = LABELS
    scores.index.name = 'Model'
    return scores.to_string(float_format=lambda value: f'{value:.3f}')


def train(args):
    import sklearn
    import torch
    import yaml
    from model import (MODEL_SPECS, CATEGORY_TO_INT, PARENT_TO_INT, encode_sequences,
                       make_folds, train_predict, empirical_library_prior, predictions_frame)

    data_path, config_path = resolve(args.data), resolve(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config['uuid'] != '6eeae50e' or not config['design_matrix']['dca'] or not config['design_matrix']['multilibrary']:
        raise ValueError('Expected the 6eeae50e DCA + multilibrary configuration.')
    params = config['training_params']
    protocol = dict(seeds=args.seeds, folds=args.folds, split_seed=20260912,
                    epochs=args.epochs if args.epochs is not None else params['num_epochs'],
                    learning_rate=params['learning_rate'], weight_decay=params['weight_decay'],
                    batch_size=params['batch_size'], hidden=[100, 50], dropout=0.2,
                    data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
                    config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256((HERE/'model.py').read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                    torch=torch.__version__, python=platform.python_version(),
                    platform=platform.platform())
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    output = resolve(args.cache_dir) / key
    output.mkdir(parents=True, exist_ok=True)
    (output/'manifest.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (output/'run.log').open('a', buffering=1) as log, redirect_stderr(log):
        warnings.simplefilter('default')
        df = pd.read_csv(data_path).reset_index().rename(columns={'index': 'row_index'})
        df['y_true'] = df.category.map(CATEGORY_TO_INT).astype(int)
        df['library'] = df.parent.map(PARENT_TO_INT).astype(int)
        if not np.isfinite(df.dca_score).all():
            raise ValueError('DCA scores must be finite.')
        fold = make_folds(df, args.folds, protocol['split_seed'])
        df.assign(cv_fold=fold)[['row_index', 'parent', 'category', 'sequence_aa_trim', 'cv_fold']].to_csv(output/'fold_assignments.csv', index=False)
        aa = torch.as_tensor(encode_sequences(df.sequence_aa_trim), dtype=torch.int64)
        dca = torch.as_tensor(df.dca_score.to_numpy(np.float32)[:, None])
        library = torch.as_tensor(df.library.to_numpy(np.int64))
        y = torch.as_tensor(df.y_true.to_numpy(np.int64))
        parts = []
        for seed in args.seeds:
            for f in range(args.folds):
                training, heldout = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
                for name, spec in MODEL_SPECS.items():
                    saved = output/f'{name}_seed{seed}_fold{f+1}.csv'
                    if saved.exists():
                        part = pd.read_csv(saved)
                    else:
                        log.write(f'Training {name}, seed={seed}, fold={f+1}\n')
                        probs, _ = train_predict(aa, dca, library, y, training, heldout, spec, seed,
                            protocol['epochs'], protocol['learning_rate'], protocol['weight_decay'],
                            protocol['batch_size'], tuple(protocol['hidden']), protocol['dropout'])
                        part = predictions_frame(df, heldout, probs, name, seed, str(f+1), 'cross_validation')
                        temp = saved.with_suffix('.tmp')
                        part.to_csv(temp, index=False)
                        temp.replace(saved)
                    parts.append(part)
                prior = empirical_library_prior(df, training, heldout)
                low = np.zeros((len(heldout), 4))
                low[:, 1] = 1
                for name, probs in [('library_prior', prior), ('always_low', low)]:
                    parts.append(predictions_frame(df, heldout, probs, name, seed, str(f+1), 'cross_validation'))
        result = pd.concat(parts, ignore_index=True)
        result.to_csv(output/'predictions.csv', index=False)
        log.write('Completed grouped CV.\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('output/ordinal_Oct22_sequences_with_dca_score.csv'))
    parser.add_argument('--config', type=Path, default=Path('output/ordinal_Oct22_models/saved_models/6eeae50e.yml'))
    parser.add_argument('--cache-dir', type=Path, default=Path('cv_balanced_accuracy/cache'))
    parser.add_argument('--predictions', type=Path, help='Print the table from an existing CV predictions.csv, without training.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, help='Override saved configuration; omit for the publication recipe.')
    parser.add_argument('--smoke', action='store_true', help='Two epochs, two folds, one seed; for testing only.')
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.folds, args.seeds = 2, 2, [100]
    if args.folds < 2 or (args.epochs is not None and args.epochs < 1) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use at least two folds, positive epochs, and distinct seeds.')
    predictions = pd.read_csv(resolve(args.predictions)) if args.predictions else train(args)
    print(table(predictions))


if __name__ == '__main__':
    main()
