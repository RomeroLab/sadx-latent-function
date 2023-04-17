import sys

import collections
import copy
from concurrent import futures
import json
import random
import shutil

# required for Jackhmmer cache
import os
from typing import Any, Callable, Mapping, Optional, Sequence

from urllib import request
#from google.colab import files
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol

from alphafold.notebooks import notebook_utils
import enum

@enum.unique
class ModelType(enum.Enum):
  MONOMER = 0
  MULTIMER = 1

from alphafold.model import model
from alphafold.model import config
from alphafold.model import data

from alphafold.data import feature_processing
from alphafold.data import msa_pairing
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data.tools import jackhmmer

from alphafold.common import protein

from alphafold.relax import relax
from alphafold.relax import utils

#from IPython import display
#from ipywidgets import GridspecLayout
#from ipywidgets import Output


## Cell 3. Enter the amino acid sequence(s) to fold
import Bio.SeqIO

input_sequences = [str(seq.seq) for seq in Bio.SeqIO.parse("target.fasta", "fasta")]

MIN_PER_SEQUENCE_LENGTH = 16
MAX_PER_SEQUENCE_LENGTH = 4000
MAX_MONOMER_MODEL_LENGTH = 2500
MAX_LENGTH = 4000
MAX_VALIDATED_LENGTH = 3000

use_multimer_model_for_monomers = False

sequences = notebook_utils.clean_and_validate_input_sequences(
    input_sequences=input_sequences,
    min_sequence_length=MIN_PER_SEQUENCE_LENGTH,
    max_sequence_length=MAX_PER_SEQUENCE_LENGTH)


if len(input_sequences) != 1:
  raise ValueError("Exactly one sequence must be specified. "
                   "Only monomers currently implemented")

model_type_to_use = ModelType.MONOMER
## END Cell 3


## Cell 4. Search against genetic databases

# required for Jackhmmer cache
from alphafold.data import parsers

JACKHMMER_BINARY_PATH = '/usr/bin/jackhmmer'
source = ""
DB_ROOT_PATH = f'https://storage.googleapis.com/alphafold-colab{source}/latest/'
# The z_value is the number of sequences in a database.
MSA_DATABASES = [
    {'db_name': 'uniref90',
     'db_path': f'{DB_ROOT_PATH}uniref90_2022_01.fasta',
     'num_streamed_chunks': 62,
     'z_value': 144_113_457},
    {'db_name': 'smallbfd',
     'db_path': f'{DB_ROOT_PATH}bfd-first_non_consensus_sequences.fasta',
     'num_streamed_chunks': 17,
     'z_value': 65_984_053},
    {'db_name': 'mgnify',
     'db_path': f'{DB_ROOT_PATH}mgy_clusters_2022_05.fasta',
     'num_streamed_chunks': 120,
     'z_value': 623_796_864},
]

# add multimeter database if needed. This is left out for now as we are only doing monomers

TOTAL_JACKHMMER_CHUNKS = sum([cfg['num_streamed_chunks'] for cfg in MSA_DATABASES])

MAX_HITS = {
    'uniref90': 10_000,
    'smallbfd': 5_000,
    'mgnify': 501,
    'uniprot': 50_000,
}


class PreCachedJackhmmer(jackhmmer.Jackhmmer):

  cache_dir="hmmer_output"

  def _query_chunk(self,
                   input_fasta_path: str,
                   database_path: str,
                   max_sequences: Optional[int] = None) -> Mapping[str, Any]:
    
    cache_path = os.path.join(self.cache_dir, database_path) 
    print(cache_path)
    sto_path = cache_path + ".sto"
    tblout_path = cache_path + ".tblout.txt"
    if not os.path.isfile(sto_path):
      raise ValueError(f"sto_path : {sto_path} cannot be found")

    # Get e-values for each target name
    tbl = ''
    if self.get_tblout:
      with open(tblout_path) as f:
        tbl = f.read()

    if max_sequences is None:
      with open(sto_path) as f:
        sto = f.read()
    else:
      sto = parsers.truncate_stockholm_msa(sto_path, max_sequences)

    stderr = ''
    raw_output = dict(
        sto=sto,
        tbl=tbl,
        stderr=stderr,
        n_iter=self.n_iter,
        e_value=self.e_value)
    return raw_output


  def query_multiple(
      self,
      input_fasta_paths: Sequence[str],
      max_sequences: Optional[int] = None,
    ) -> Sequence[Sequence[Mapping[str, Any]]]:
    """ Intercept query_multiple from base class and return pre-cached results"""
    if self.num_streamed_chunks is None:
      raise ValueError("num_streamed_chunks cannot be None as we only cache chunks")
    db_basename = os.path.basename(self.database_path)

    db_local_chunk = lambda db_idx: f'{db_basename}.{db_idx}'

    chunked_outputs = [[] for _ in range(len(input_fasta_paths))]

    for i in range(1, self.num_streamed_chunks + 1):
      for fasta_index, input_fasta_path in enumerate(input_fasta_paths):
        chunked_outputs[fasta_index].append(self._query_chunk(
          input_fasta_path, db_local_chunk(i), max_sequences))

    return chunked_outputs
  

def get_msa(sequences):
  sequence_to_fasta_path = {}
  for sequence_index, sequence in enumerate(sorted(set(sequences)), 1):
    if sequence_index > 1:
      raise ValueError("Only one sequence supported")
    fasta_path = f'target_{sequence_index:02d}.fasta'
    with open(fasta_path, 'wt') as f:
      f.write(f'>query\n{sequence}')
    sequence_to_fasta_path[sequence] = fasta_path
  raw_msa_results = {sequence: {} for sequence in sequence_to_fasta_path.keys()}


  jackhmmer_chunk_callback = lambda x: None
  for db_config in MSA_DATABASES:
    db_name = db_config['db_name']
    jackhmmer_runner = PreCachedJackhmmer(
      binary_path=JACKHMMER_BINARY_PATH,
      database_path=db_config['db_path'],
      get_tblout=True,
      num_streamed_chunks=db_config['num_streamed_chunks'],
      streaming_callback=jackhmmer_chunk_callback,
      z_value=db_config['z_value'])
    results = jackhmmer_runner.query_multiple(list(sequence_to_fasta_path.values()))
    for sequence, result_for_sequence in zip(sequence_to_fasta_path.keys(), results):
      raw_msa_results[sequence][db_name] = result_for_sequence

  return raw_msa_results

features_for_chain = {}
raw_msa_results_for_sequence = get_msa(sequences)

for sequence_index, sequence in enumerate(sequences, start=1):
  raw_msa_results = copy.deepcopy(raw_msa_results_for_sequence[sequence])

  # Extract the MSAs from the Stockholm files.
  # NB: deduplication happens later in pipeline.make_msa_features.
  single_chain_msas = []
  uniprot_msa = None
  for db_name, db_results in raw_msa_results.items():
    merged_msa = notebook_utils.merge_chunked_msa(
        results=db_results, max_hits=MAX_HITS.get(db_name))
    if merged_msa.sequences and db_name != 'uniprot':
      single_chain_msas.append(merged_msa)
      msa_size = len(set(merged_msa.sequences))
      print(f'{msa_size} unique sequences found in {db_name} for sequence {sequence_index}')
    elif merged_msa.sequences and db_name == 'uniprot':
      uniprot_msa = merged_msa

  notebook_utils.show_msa_info(single_chain_msas=single_chain_msas, sequence_index=sequence_index)

  # Turn the raw data into model features.
  feature_dict = {}
  feature_dict.update(pipeline.make_sequence_features(
      sequence=sequence, description='query', num_res=len(sequence)))
  feature_dict.update(pipeline.make_msa_features(msas=single_chain_msas))
  # We don't use templates in AlphaFold Colab notebook, add only empty placeholder features.
  feature_dict.update(notebook_utils.empty_placeholder_template_features(
      num_templates=0, num_res=len(sequence)))

  # Construct the all_seq features only for heteromers, not homomers.
  if model_type_to_use == ModelType.MULTIMER and len(set(sequences)) > 1:
    valid_feats = msa_pairing.MSA_FEATURES + (
        'msa_species_identifiers',
    )
    all_seq_features = {
        f'{k}_all_seq': v for k, v in pipeline.make_msa_features([uniprot_msa]).items()
        if k in valid_feats}
    feature_dict.update(all_seq_features)

  features_for_chain[protein.PDB_CHAIN_IDS[sequence_index - 1]] = feature_dict

# Do further feature post-processing depending on the model type.
if model_type_to_use == ModelType.MONOMER:
  np_example = features_for_chain[protein.PDB_CHAIN_IDS[0]]

elif model_type_to_use == ModelType.MULTIMER:
  all_chain_features = {}
  for chain_id, chain_features in features_for_chain.items():
    all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
        chain_features, chain_id)

  all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)

  np_example = feature_processing.pair_and_merge(
      all_chain_features=all_chain_features)

  # Pad MSA to avoid zero-sized extra_msa.
  np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=512)
## END Cell 4

## Cell 5. Run AlphaFold and download prediction
# testing

run_relax = False
relax_use_gpu = False
multimer_model_max_num_recycles = 3  

if model_type_to_use == ModelType.MONOMER:
  model_names = config.MODEL_PRESETS['monomer'] + ('model_2_ptm',)
elif model_type_to_use == ModelType.MULTIMER:
  model_names = config.MODEL_PRESETS['multimer']

output_dir = 'prediction'
os.makedirs(output_dir, exist_ok=True)

#model_name = "model_1"
#params = data.get_model_haiku_params(model_name, '../alphafold/data')

for model_name in model_names:
  print(f'Running {model_name}')

  cfg = config.model_config(model_name)

  if model_type_to_use == ModelType.MONOMER:
    cfg.data.eval.num_ensemble = 1
  elif model_type_to_use == ModelType.MULTIMER:
    cfg.model.num_ensemble_eval = 1

  if model_type_to_use == ModelType.MULTIMER:
    cfg.model.num_recycle = multimer_model_max_num_recycles
    cfg.model.recycle_early_stop_tolerance = 0.5

  params = data.get_model_haiku_params(model_name, '../alphafold/data')
  model_runner = model.RunModel(cfg, params)
  processed_feature_dict = model_runner.process_features(np_example, random_seed=0)
  prediction = model_runner.predict(processed_feature_dict, random_seed=random.randrange(sys.maxsize))

  mean_plddt = prediction['plddt'].mean()

  if model_type_to_use == ModelType.MONOMER:
    if 'predicted_aligned_error' in prediction:
      pae_outputs[model_name] = (prediction['predicted_aligned_error'],
                                 prediction['max_predicted_aligned_error'])
    else:
      # Monomer models are sorted by mean pLDDT. Do not put monomer pTM models here as they
      # should never get selected.
      ranking_confidences[model_name] = prediction['ranking_confidence']
      plddts[model_name] = prediction['plddt']
  elif model_type_to_use == ModelType.MULTIMER:
    # Multimer models are sorted by pTM+ipTM.
    ranking_confidences[model_name] = prediction['ranking_confidence']
    plddts[model_name] = prediction['plddt']
    pae_outputs[model_name] = (prediction['predicted_aligned_error'],
                               prediction['max_predicted_aligned_error'])
  
  # Set the b-factors to the per-residue plddt.
  final_atom_mask = prediction['structure_module']['final_atom_mask']
  b_factors = prediction['plddt'][:, None] * final_atom_mask
  unrelaxed_protein = protein.from_prediction(
      processed_feature_dict,
      prediction,
      b_factors=b_factors,
      remove_leading_feature_dimension=(
          model_type_to_use == ModelType.MONOMER))
  unrelaxed_proteins[model_name] = unrelaxed_protein
  
  # Delete unused outputs to save memory.
  del model_runner
  del params
  del prediction

# Find the best model according to the mean pLDDT.
best_model_name = max(ranking_confidences.keys(), key=lambda x: ranking_confidences[x])

if run_relax:
  print(f'AMBER relaxation')
  amber_relaxer = relax.AmberRelaxation(
      max_iterations=0,
      tolerance=2.39,
      stiffness=10.0,
      exclude_residues=[],
      max_outer_iterations=3,
      use_gpu=relax_use_gpu)
  relaxed_pdb, _, _ = amber_relaxer.process(prot=unrelaxed_proteins[best_model_name])
else:
  print('Warning: Running without the relaxation stage.')
  relaxed_pdb = protein.to_pdb(unrelaxed_proteins[best_model_name])

# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
banded_b_factors = []
for plddt in plddts[best_model_name]:
  for idx, (min_val, max_val, _) in enumerate(PLDDT_BANDS):
    if plddt >= min_val and plddt <= max_val:
      banded_b_factors.append(idx)
      break
banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
to_visualize_pdb = utils.overwrite_b_factors(relaxed_pdb, banded_b_factors)

# Write out the prediction
pred_output_path = os.path.join(output_dir, 'selected_prediction.pdb')
with open(pred_output_path, 'w') as f:
  f.write(relaxed_pdb)

## END Cell 5
