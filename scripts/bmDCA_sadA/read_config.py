"""
Python file to read config file. 

This file should work with any python3 version as it needs to be run
outside this repo. 
"""
import functools
import pathlib

import yaml

class Config:
    """Miscellaneous properties for config, derived from a config.yaml file

    To use:
    >>> config = Config(config_yaml_filename)
    >>> config.aligned_msa_fullpath
    PosixPath('../sequence_sets/cmx_aligned_blank_90.fasta')
    >>> config.reweighting_threshold
    0.8
    >>> config.weights_fullpath # combining properties
    PosixPath('../working/cmx_aligned_blank_90.npy')
    """

    # where the root directory is relative to this file
    root_dir = pathlib.Path("..")
    
    def __init__(self, yaml_filename):
        self.data =  self._load_yaml(yaml_filename)

    def _load_yaml(self, yaml_filename):
        """ Simple function to load yaml file """
        with open(yaml_filename, 'r') as fh:
            yaml_data = yaml.safe_load(fh)
        return yaml_data

    def __str__(self):
        """Search for properties of this class and return a string consisting
        of a sorted list of properties and values""" 
        properties = [attr for attr in dir(self) 
                        if isinstance(getattr(Config,attr, None),property)]
        properties.sort()
        pstr_list = ["{0}: {1}".format(p, getattr(self, p)) for p in properties]
        return "\n".join(pstr_list)

    def safe_get_key(self, key, default="", print_if_not_found=True):
        """Get a key from the yaml file but if it doesn't exist then set it to
            default
        """
        ret = default
        try:
            ret = self.data[key]
        except KeyError:
            if print_if_not_found:
                print(f"Cannot get : {key}, defaulting to : {default}")
            pass
        return (ret)

    @property
    def aligned_msa_filename(self):
        """String input msa filename from yaml file"""
        return self.safe_get_key('aligned_msa_filename')

    @property
    def reweighting_threshold(self):
        """Float re-weighting threshold (similarity to a sequence)"""
        return float(self.safe_get_key('reweighting_threshold', 0.8))

    @property
    def aligned_msa_fullpath(self):
        """Complete path to the input MSA"""
        return self.dataset_dir / self.aligned_msa_filename

    @property
    def weights_fullpath(self):
        """Complete path to the output weights npy file"""
        msa_filename_stem = pathlib.Path(self.aligned_msa_filename
                                            ).stem.split('.')[0]
        return (self.working_dir / pathlib.Path(msa_filename_stem + 
                    "_weights")).with_suffix('.npy')

    @property
    def dataset_dir(self):
        """Complete path to the dataset directory """
        return Config.root_dir / pathlib.Path(self.safe_get_key('dataset_dir', 
                'sequence_sets'))

    @property
    def working_dir(self):
        """ Complete path to the working directory """
        return Config.root_dir / pathlib.Path(self.safe_get_key('working_dir', 
                'sequence_sets'))

    @property
    def hidden_layer_size(self):
        """ Number of nodes in hidden layer """
        return self.safe_get_key('hidden_layer_size')

    @property
    def latent_layer_size(self):
        """ Number of nodes in latent layer """
        return self.safe_get_key('latent_layer_size')

    @property
    def activation_function(self):
        """ Type of activation function. Options: Sigmoid, """
        function = self.safe_get_key('activation_function')
        ret = None
        if function == 'sigmoid':
            from torch import nn
            ret = nn.Sigmoid()
        return ret

    @property
    def learning_rate(self):
        """ Parameter to scale gradient descent updates """
        return self.safe_get_key('learning_rate')

    @property
    def epochs(self):
        """ Number of epochs to train model """
        return self.safe_get_key('epochs')

    @property
    def batch_size(self):
        """ Get batch_size"""
        return self.safe_get_key('batch_size')

    @property
    def model_name(self):
        """ name of file to save model """
        return self.safe_get_key('model_name')

    @property
    def device(self):
        """ device to run the model on"""
        device = self.safe_get_key('device', '')
        if device == '' or device == 'auto':
            device = get_best_device()
        return device

    @property
    def model_fullpath(self):
        """Complete path to the saved model"""
        return self.working_dir / \
                pathlib.Path(self.model_name).with_suffix(".pt")

    @property
    def latent_fullpath(self):
        """Complete path to the saved latent"""
        return self.working_dir / \
                pathlib.Path(self.model_name + "_latent.pkl.gz")

    @property
    def loss_fullpath(self):
        """Complete path to the saved loss"""
        return self.working_dir / \
                pathlib.Path(self.model_name + "_loss").with_suffix(".pkl")

    @property
    def lossgraph_fullpath(self):
        return self.working_dir / \
                pathlib.Path(self.model_name + "_loss").with_suffix(".png")

    @property
    def foreground_sequences_filename(self):
        return self.safe_get_key('foreground_sequences_filename', '')

    @property
    def foreground_sequences_fullpath(self):
        return self.dataset_dir / \
                pathlib.Path(self.foreground_sequences_filename)

    @property
    def foreground_sequences_output_fullpath(self):
        return self.working_dir / \
                pathlib.Path(
                        self.latent_plot_output_filename
                        ).with_suffix(".png")

    @property
    def foreground_sequences_label(self):
        return self.safe_get_key('foreground_sequences_label',
                default= str(self.foreground_sequences_fullpath.stem))

    @property
    def latent_plot_output_filename(self):
        return self.safe_get_key('latent_plot_output_filename', 
                default=self.model_name + "_latent.png")

    @property
    def latent_plot_output_fullpath(self):
        return self.working_dir / pathlib.Path(
                        self.latent_plot_output_filename
                        ).with_suffix(".png")

    @property
    def latent_plot_archive_fullpath(self):
        return self.latent_plot_output_fullpath.with_suffix(".pkl")

    @property
    def dca_regularization(self):
        """ Options are l2 or l1 """
        return self.safe_get_key('dca_regularization', default="l2")

    @property
    def dca_params_filename(self):
        return self.safe_get_key('dca_params_filename', 
                default=self.model_name + "_dca_params.pkl")

    @property
    def dca_params_fullpath(self):
        return self.working_dir / pathlib.Path(
                        self.dca_params_filename,
                        ).with_suffix(".pkl")

    @property
    def convert_unknown_aa_to_gap(self):
        return self.safe_get_key('convert_unknown_aa_to_gap', False, 
                print_if_not_found=False)

    @property
    def log_level(self):
        return self.safe_get_key('log_level', "INFO", 
                print_if_not_found=False)

def get_best_device():
    """Get the device to use for pytorch

       Return gpu device if the gpu is available else return cpu
    """
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process yaml config file')
    parser.add_argument("config_filename",
                    help="input config file in yaml format")

    # All the property names are arguments
    # If any one is specified we will print out that property only
    property_names=[p for p in dir(Config) if 
                    isinstance(getattr(Config,p), property)]
    for p in property_names:
        parser.add_argument(f"--{p}", action='store_true')

    args = parser.parse_args()

    config = Config(args.config_filename)

    no_args_printed = True
    for arg in vars(args):
        if getattr(args, arg) and (arg in property_names):
            print(getattr(config, arg))
            no_args_printed = False
            break

    if no_args_printed: # print entire config file
        try:
            print(config)
        except ImportError: # probably an error import torch on chtc
            print("Error: If running on CHTC then provide a property to print")

