


### Create list of random mutants

```shell
for i in {1..9}; do ./generate_variants.py -p WT -d $i -n 100; done > variant_list.txt
```
