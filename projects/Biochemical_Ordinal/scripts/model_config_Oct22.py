import hashlib
import yaml

class ModelConfig:
    """ Save some info about a model as a yaml file so that we can analyze
        results later"""

    @classmethod
    def create_from_yaml(cls, filename):
        with open(filename, "r") as fh:
            x = yaml.safe_load(fh)
            # put the design matrix values at the top level
            # as they are sent directly to the constructor of this class
            for key in ["design_matrix", "model_params", "dataset_params"]:
                dm = x.pop(key, {})
                for k, v in dm.items():
                    x[k] = v
            return cls(**x)

    @classmethod
    def create_from_args(cls, args):
        kwargs = {}
        for key in ['train_name', 'val_name', 'test_name',
                    'intercept']:
            if key in kwargs:
                kwargs[key] = args[key]
        return cls(
                model_name = args.model_name,
                target = args.target,
                dca = args.dca,
                encoding = args.encoding,
                **kwargs)
   
    def __init__(self, model_name, 
                 target="multiclass", # binary or multiclass 
                 intercept=False, # whether the design matrix has an intercept or not
                 dca=False,  # whether dca is added to design matrix
                 multilibrary=False, # whether we consider separate libraries
                 encoding="one-hot", # one-hot, esm, esmft
                 uuid=None,
                 seed=100,
                 train_name='train',
                 val_name='',
                 test_name='test',
                 **model_params
                ):
      
        self.model_name = model_name
        
        self.target = target

        self.dataset_params = {'train_name':train_name,
                               'val_name':val_name,
                               'test_name':test_name}
        
        self.design_matrix = {'intercept':intercept,
                             'dca':dca,
                             'multilibrary':multilibrary,
                             'encoding':encoding}

        # add additional arguments
        self.model_params = model_params.copy()

        self.seed = seed

        self.uuid = uuid
        if self.uuid is None:
            self.uuid = self.build_uuid()
            
    def get_config_as_dict(self, with_uuid=True):
        ret = {
               'model_name':self.model_name, 
               'target':self.target,
               'design_matrix':self.design_matrix, 
               'model_params':self.model_params,
               'dataset_params':self.dataset_params
               }      
        if with_uuid:
            ret["uuid"] = self.uuid
        
        return ret
      
    def build_uuid(self):
        """ hash a text representation of this classes dict"""
        variables_d = self.get_config_as_dict(with_uuid=False)
        return hashlib.blake2b(bytes(variables_d.__repr__(), "utf-8"), 
                             digest_size=4).hexdigest()
    
    def __repr__(self):
        return self.save_yaml(write_to_file=False)
    
    def get_default_filename(self):
        return f"{self.uuid}.yml"
      
    def save_yaml(self, write_to_file=True, filename=None, directory=None):
        """Directory should be pathlib if provided"""
        fh = None
        if write_to_file:
            if filename is None:
              filename = self.get_default_filename()
            if directory is not None:
              filename = directory / filename
            fh = open(filename, "w")
        ret = yaml.dump(self.get_config_as_dict(), stream=fh)
        if fh is not None:
            fh.close()
        if write_to_file:
             ret = None 
        return ret

if __name__ == "__main__":
    x = ModelConfig("LogisticRegression")
    print(x)
