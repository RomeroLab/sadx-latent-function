import hashlib
import yaml


class ModelConfig:
    """ Save some info about a model as a yaml file so that we can analyze
        results later"""
  
    def __init__(self, model_name, 
                 target="multiclass", # binary or multiclass 
                 intercept=False, # whether the design matrix has an intercept or not
                 dca=False,  # whether dca is added to design matrix
                 multilibrary=False, # whether we consider separate libraries
                 encoding="one-hot", # one-hot, esm, esmft
                 uuid=None,
                 seed=100,
                 **model_params
                ):
      
        self.model_name = model_name
        
        self.target = target
        
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
        ret = {'model_name':self.model_name, 'target':self.target,
                      'design_matrix':self.design_matrix, 
                      'model_params':self.model_params}      
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
