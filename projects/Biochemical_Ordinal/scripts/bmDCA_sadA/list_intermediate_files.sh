

# find the latest saved params (using timestamps)
latest_file=`ls -t results/parameters_J*.bin | head -1`
latest_iter_num=`basename -s .bin "$latest_file" | awk -F_ '{print $3}'`

# extended globbing should be on 
shopt -s extglob

# list everything that ends in one or more numbers
ls results/*_+([0-9]).* | \
        awk -v l=$latest_iter_num '{
            fn=$NF; 
            sub(/.bin/, "", fn); 
            sub(/.txt/, "", fn); 
            sub(/^.*_/, "", fn); 
            if ( fn~/^[0-9]+/ && fn+0 != l ) 
                print $0
            }' 

