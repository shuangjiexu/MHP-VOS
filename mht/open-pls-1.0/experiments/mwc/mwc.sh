#! /bin/bash

time_out=10
seeds=1
seeds_minus_one=$((seeds - 1))

output_dir=results/mwc.$seeds.$time_out
data_dir=$1

#mv $output_dir  old.$output_dir

mkdir $output_dir

for file_name in `ls -1 $data_dir`; do

    data_name=`echo $file_name | sed -e "s/-sorted//g" | sed -e "s/.graph//g"`
    target_weight=`cat known.mwc | grep $data_name | awk '{ print $1}'`

    log_file=$output_dir/log.$file_name.$time_out

    rm -rf $log_file

    echo "Target weight=$target_weight"
    echo -n "Running $file_name "
    for random_seed in $(seq 0 $seeds_minus_one); do
        echo -n "$random_seed/$seeds_minus_one..."
        #echo "../../bin/pls --input-file=$data_dir/$file_name --weighted --target-weight=$target_weight --timeout=$time_out --random-seed=$random_seed >> $log_file"
        ../../bin/pls --input-file=$data_dir/$file_name --weighted --target-weight=$target_weight --timeout=$time_out --random-seed=$random_seed >> $log_file
    done
    echo ""
    echo "Add to table..."
    python tablegen.py $output_dir/log.$file_name.$time_out >> $output_dir/mwc.table
done


### generate and open table

cat header.mwc $output_dir/mwc.table footer.mwc > $output_dir/mwc.table.tex
cd $output_dir
pdflatex mwc.table.tex
pdflatex mwc.table.tex
open -a Skim mwc.table.pdf
