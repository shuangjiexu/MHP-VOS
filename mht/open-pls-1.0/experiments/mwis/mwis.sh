#! /bin/bash

time_out=10
seeds=10
seeds_minus_one=$((seeds - 1))

output_dir=results/mwis.$seeds.$time_out
data_dir=$1

#mv $output_dir  old.$output_dir

mkdir $output_dir
rm -rf $output_dir/log.*

for file_name in `ls -1 $data_dir/ | grep "\.graph$"`; do

    data_name=`echo $file_name | sed -e "s/-sorted//g" | sed -e "s/.graph//g"`
    target_weight=`cat known.mwis | grep $data_name | awk '{ print $1}'`

    log_file=$output_dir/log.$file_name.$time_out

    rm -rf $log_file

    echo "Target weight=$target_weight"
    echo -n "Running $file_name "
    for random_seed in $(seq 0 $seeds_minus_one); do
        echo -n "$random_seed/$seeds_minus_one..."
        ../../bin/pls --algorithm=mwis --input-file=$data_dir/$file_name --target-weight=$target_weight --weighted --use-weight-file --timeout=$time_out --random-seed=$random_seed >> $log_file
    done
    echo ""
    echo "Add to table..."
    python tablegen.py $output_dir/log.$file_name.$time_out >> $output_dir/mwis.table
done


### generate and open table

cat header.mwis $output_dir/mwis.table footer.mwis > $output_dir/mwis.table.tex
cd $output_dir
pdflatex mwis.table.tex
pdflatex mwis.table.tex
open -a Skim mwis.table.pdf
