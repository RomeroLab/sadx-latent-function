


### Create list of random mutants

```shell
for i in {1..9}; do ./generate_variants.py -p WT -d $i -n 100; done > variant_list.txt
```

Create list of mutants that were in the 2D EPPCR dataset
```shell
less ../data/2D/azid_model_sequences.csv  \
    | awk -F, '{print $1}' \
    | sed 's/\;/\./g' \
    | grep . \
    | sed '1d' > variant_3D_EPPCR.txt
```
