#! /bin/bash

time_out=5
seeds=10
seeds_minus_one=$((seeds - 1))

output_dir=results/mc.$seeds.$time_out
data_dir=$1

#mv $output_dir  old.$output_dir

mkdir $output_dir

for file_name in `ls -1 $data_dir`; do

    data_name=`echo $file_name | sed -e "s/-sorted//g" | sed -e "s/.graph//g"`
    target_weight=`cat known.mc | grep $data_name | awk '{ print $1}'`

    log_file=$output_dir/log.$file_name.$time_out

    rm -rf $log_file

    echo "Target weight=$target_weight"
    echo -n "Running $file_name "
    for random_seed in $(seq 0 $seeds_minus_one); do
        echo -n "$random_seed/$seeds_minus_one..."
        #echo "../../bin/pls --input-file=$data_dir/$file_name --target-weight=$target_weight --timeout=$time_out --random-seed=$random_seed >> $log_file"
        ../../bin/pls --input-file=$data_dir/$file_name --target-weight=$target_weight --timeout=$time_out --random-seed=$random_seed >> $log_file
    done
    echo ""
    echo "Add to table..."
    python tablegen.py $output_dir/log.$file_name.$time_out >> $output_dir/mc.table
done


### generate and open table

cat header.mc $output_dir/mc.table footer.mc > $output_dir/mc.table.tex
cd $output_dir
pdflatex mc.table.tex
pdflatex mc.table.tex
open -a Skim mc.table.pdf
