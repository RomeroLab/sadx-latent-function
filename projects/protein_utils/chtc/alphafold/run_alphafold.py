

# required from Jackhmmer cache
from typing import Any, Callable, Mapping, Optional, Sequence
from alphafold.data import parsers
import os



import collections
import copy
from concurrent import futures
import json
import random
import shutil

from urllib import request
#from google.colab import files
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
#import py3Dmol

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


# testing
model_name = "model_1"
params = data.get_model_haiku_params(model_name, '../alphafold/data')
