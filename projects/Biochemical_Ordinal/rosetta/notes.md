

### Create list of 100K triple mutants from Parent 2D
```
# get unique variants. Very unlikely to repeat variants at distance=3
# but we are doing this in case we use the same command at distance 1 or 2
./generate_variants.py -p PARENT_2D -d 3 -n 100000 | sort  | uniq | shuf >  variant_list_2D_random.txt
# split variant list 
split -l 9000 variant_list_2D_random.txt variant_list_2D_random_ --additional-suffix=.txt
```


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
