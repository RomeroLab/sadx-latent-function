import torch 

class CoralMultipleLayer(torch.nn.Module):
    """ Implements CORAL layer described in
    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).
    num_classes : int
        Number of classes in the dataset.
    preinit_bias : bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.
    """
    def __init__(self, size_in, num_classes, num_datasets=1, 
                            preinit_bias=False):
        super().__init__()
        self.size_in, self.size_out = size_in, 1
        self.num_classes = num_classes

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        #FIXME: CHECK the shapes of these arrays that they are reshaped
        # correctly
        #self.coral_bias = torch.nn.Identity()
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1)
                    .float()
                    .unsqueeze(1)
                    .repeat(1, num_datasets) / (num_classes - 1))
        else:
            self.coral_bias = torch.nn.Parameter(
                torch.zeros((num_classes - 1)*num_datasets)
                    .float()).reshape(num_datasets, num_classes-1)

    def forward(self, x, dx):
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        #print("x.shape=", x.shape, "dx.shape=", dx.shape)
        a = self.coral_weights(x) 
        #print("self.coral_bias.shape=",self.coral_bias.shape)
        b = torch.index_select(self.coral_bias, 0, dx)
        #print("a.shape=", a.shape, "b.shape=", b.shape)
        return a+b


