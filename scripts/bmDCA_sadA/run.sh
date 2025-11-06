
# run in scratch directory
mkdir -p run

bmdca \
  -i ~/sameerd/projects/Biochemical_Ordinal/data/msa/sadA_clean.fasta \
  -r -d ./run \
  -c bmdca.conf

# find out last run number and then 
# convert params to text
arma2ascii -p run/parameters_h_200.bin -P run/parameters_J_200.bin 

# convert params to numpy
~/sameerd/projects/protein_utils/bin/convert_bmDCA.sh \
        -i run/parameters_200.txt \
        -o sadA
