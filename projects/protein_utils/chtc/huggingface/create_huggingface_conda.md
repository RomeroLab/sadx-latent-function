
### Creating the conda environment for ESM models to run in

The challenge here was to install everything in a way that `conda pack` did not complain.

```shell
## Remove old environment if we are not starting clean
#conda env remove -n huggingface

conda create -qy -n huggingface \
        --channel huggingface \
        transformers pytorch

# activate the environment
conda activate huggingface

conda install pytorch torchvision torchaudio \
        pytorch-cuda=11.8 -c pytorch -c nvidia

pip install ipython
```

## Installing ESM-2

Following the huggingface documentation [here](https://huggingface.co/docs/transformers/model_doc/esm). 

Download a small version to cache
```shell
HUGGINGFACE_HUB_CACHE=~/hf_cache python3 - <<'EOF'
from transformers import AutoTokenizer, EsmModel
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
EOF
```

Try to run 
```shell
HUGGINGFACE_HUB_CACHE=~/hf_cache TRANSFORMERS_OFFLINE=1 python3 - <<'EOF'
from transformers import AutoTokenizer, EsmModel
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
EOF
```
