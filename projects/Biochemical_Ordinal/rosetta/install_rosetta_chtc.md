## Putting the Rosetta installation on chtc

### Prepare files on group server 

```shell
# make directory to store the tmp files
export ROSETTA_SQUID_COPY="/mnt/scratch/sameer/rosetta/squid"
mkdir -p ${ROSETTA_SQUID_COPY}
cd ${ROSETTA_SQUID_COPY}

export ROSETTA_MAIN_DIR="/mnt/scratch/software/rosetta/rosetta.binary.linux.release-296/main"

# archive the Rosetta database
# split into chunks of 1000MB so they are below 1GB squid limit
(cd $ROSETTA_MAIN_DIR && tar -cf - database) \
    | bzip2 -c \
    | split -b 1000M - "db.tar.bz2.part" 
```

### Prepare directory on CHTC
```shell
mkdir -p /squid/dcosta2/2021_38 # Rosetta version number 2021.38
```

### Copy files from group server to CHTC
```shell
cd ${ROSETTA_SQUID_COPY}
cp ${ROSETTA_MAIN_DIR}/source/bin/rosetta_scripts.static.linuxgccrelease .
cp ${ROSETTA_MAIN_DIR}/source/bin/relax.static.linuxgccrelease .

# copy over parts of the database and the binary
scp db.tar.bz2.part* *.static.linuxgccrelease dcosta2@transfer.chtc.wisc.edu:/squid/dcosta2/2021_38

# create a local database copy
cat db.tar.bz2.part* | tar -jx

# delete database parts
/bin/rm -f db.tar.bz2.parta?
```
