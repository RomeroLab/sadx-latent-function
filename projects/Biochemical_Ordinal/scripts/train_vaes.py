import numpy as np

import train_vaes_phils_nb as vaes_nb
import pytorch_lightning as pl

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_MSA(filename):
    return [''.join(s.strip().split('\n')[1:]) for s in 
                open(filename).read().strip().split('>')][1:]


if __name__ == "__main__":
    MSA = load_MSA("../data/sadA_clean.fasta")
    MSA_weights = np.load("../data/sadA_clean.weights.npy")

    # load common aspects between models
    dm = vaes_nb.ProtDataModule(MSA,64)
    slen = len(MSA[0])

    ## train ConvVAE model 
    ## init model
    #kernel_size = 10
    #nlatent = 10
    #learning_rate=0.001
    #model = ConvVAE(slen, kernel_size, nlatent,learning_rate)

    ## train 
    #logger = CSVLogger('logs', name='CVAE')
    #trainer = pl.Trainer(logger=logger,max_epochs=15,gpus=1)
    #trainer.fit(model,dm)
    ##trainer.save_checkpoint("trained_VAE_model.ckpt")

    # train DCA model 
    # init model
    learning_rate=0.001
    model = vaes_nb.ProtDCA(slen,learning_rate)

    # train 
    logger = vaes_nb.CSVLogger('logs', name='DCA')
    trainer = pl.Trainer(logger=logger,max_epochs=15,gpus=0)
    trainer.fit(model,dm)
    #trainer.save_checkpoint("trained_DCA_model.ckpt")

    metrics = pd.read_csv(f"{logger.log_dir}/metrics.csv")
    metrics_melt = pd.melt(metrics, id_vars=["epoch", "step"]).dropna()
    g = sns.lineplot(x="epoch", y="value", hue="variable", data=metrics_melt)
    g.get_figure().savefig(f"{logger.log_dir}/losses.png", dpi=300)
