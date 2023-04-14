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

#from alphafold.relax import relax
#from alphafold.relax import utils

#from alphafold.relax import relax
#from alphafold.relax import utils

#from IPython import display
#from ipywidgets import GridspecLayout
#from ipywidgets import Output

# testing
#model_name = "model_1"
#params = data.get_model_haiku_params(model_name, '../alphafold/data')
