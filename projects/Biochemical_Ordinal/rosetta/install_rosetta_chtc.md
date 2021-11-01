## Putting the Rosetta installation on chtc

### Prepare files on group server 

```shell
# make directory to store the tmp files
mkdir -p /mnt/scratch/sameer/tmp
cd /mnt/scratch/sameer/tmp

# split into chunks of 1000MB so they are below 1GB squid limit
(cd /mnt/scratch/software/rosetta/rosetta_bin_linux_2019.35.60890_bundle/main \
    && tar -cf - database) \
    | bzip2 -c \
    | split -b 1000M - "db.tar.bz2.part" 


```

### Prepare directory on CHTC
```shell
cd /squid/dcosta2
mkdir 3_10 #version number
```

### Copy files from group server to CHTC
```shell
cd /mnt/scratch/sameer/tmp

scp db.tar.bz2.part* dcosta2@transfer.chtc.wisc.edu:/squid/dcosta2/3_10
scp /mnt/scratch/software/rosetta/rosetta_bin_linux_2019.35.60890_bundle/main/source/build/src/release/linux/3.10/64/x86/gcc/4.8/static/rosetta_scripts.static.linuxgccrelease dcosta2@transfer.chtc.wisc.edu:/squid/dcosta2/3_10


```
