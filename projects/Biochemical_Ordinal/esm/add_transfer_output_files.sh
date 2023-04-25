echo -n "transfer_output_files="
current_line=0
while IFS= read -r line; 
do
    if [[ $current_line -ne 0 ]]; then
        echo -n ,
    fi
    echo -n ${line}
    current_line=$(($current_line + 1))
done < "$1"
