import pathlib
import numpy as np

DEFAULT_AA_PCA_FILE=pathlib.Path(__file__).parents[1] / "data/AA/pca-19.csv"

def get_aaindex_embedding(filename=DEFAULT_AA_PCA_FILE, ncomp = 8):
    # get aa index embedding
    aaindex = np.array([[float(f) for f in l.split(',')[1:]] for l in
            open(filename).read().strip().split('\n')[1:]])
    aaindex = (aaindex - aaindex.mean(0))/aaindex.std(0) # standardize
    # add final row to include gap -
    aaindex = np.vstack([aaindex,np.zeros((1,19))]) 
    #aaindex = torch.from_numpy(aaindex).float()
    aaindex = aaindex[:,:ncomp]
    return aaindex
