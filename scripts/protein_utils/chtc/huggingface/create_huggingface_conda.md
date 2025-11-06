
### Creating the conda environment for ESM models to run in

The challenge here was to install everything in a way that `conda pack` did not complain.

```shell
## Remove old environment if we are not starting clean
#conda env remove -n huggingface

conda create -qy -n huggingface pytorch torchvision torchaudio \
        pytorch-cuda=11.8 transformers -c pytorch -c nvidia -c huggingface

# activate the environment
conda activate huggingface

pip install ipython

conda deactivate
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

## Make conda environment
```shell
conda pack -n huggingface
tar tvf huggingface.tar.gz

du -sh huggingface.tar.gz



```
