

latest_file=`ls results/parameters_J*.bin | tail -1`
latest_iter_num=`basename -s .bin "$latest_file" | awk -F_ '{print $3}'`

ls results/*_[0-9]*.* | awk -v l=$latest_iter_num '{fn=$NF; sub(/.bin/, "", fn); sub(/^.*_/, "", fn); if ( fn~/^[0-9]+/ && fn+0 != l ) print $0}' 

