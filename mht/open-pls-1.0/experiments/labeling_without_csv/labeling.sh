#! /bin/bash

time_out=0.1
seeds=1
seeds_minus_one=$((seeds - 1))

output_dir=results/labeling.$seeds.$time_out
data_dir=$1
csv_dir=$output_dir/csvs

#mv $output_dir  old.$output_dir

mkdir $output_dir
rm -rf $output_dir/log.*
rm -rf $csv_dir
mkdir $csv_dir

echo "SEED,K,AM,PLS MAX SCORE,PLS MAX TIME" > $output_dir/intermediate.csv

for file_name in `ls -1 $data_dir/ | grep "\.graph$"`; do

# graph information about data set, AM1 or AM2 or AM3

    data_name=`echo $file_name | sed -e "s/.graph$//g"`
    echo "data_name=$data_name"

    graphml_file_name=$data_dir/../graphml/$data_name.graphml
    echo "graphml_file_name=$graphml_file_name"

    node_mapping_file="$data_dir/$file_name.node_mapping"

    am=`echo $file_name | sed -e "s/.*AM\([1-3]\).*/\1/g"`
    echo "am=AM$am"

    prefix=`echo $file_name | sed -e "s/\(.*\)\.graphml.*/\1/g"`
    echo "prefix=$prefix"

    seed=`echo $prefix | sed -e "s/.*seed_\([0-9]*\).*/\1/g"`
    echo "seed=$seed"

    k=`echo $prefix | sed -e "s/.*seed_[0-9]*-\(.*\)/\1/g"`
    echo "k=$k"

    log_file=$output_dir/log.$file_name.timeout_$time_out

    #rm -rf $log_file

    echo -n "Running $file_name "
    for random_seed in $(seq 0 $seeds_minus_one); do
        echo -n "$random_seed/$seeds_minus_one..."
        echo "../../bin/pls --algorithm=labeling --input-file=$data_dir/$file_name --weighted --use-weight-file --timeout=$time_out --random-seed=$random_seed >> $log_file"
        ../../bin/pls --algorithm=labeling --input-file=$data_dir/$file_name --weighted --use-weight-file --timeout=$time_out --random-seed=$random_seed >> $log_file
        #echo "greedy-weight: $greedy_weight" >> $log_file
        echo "file-seed: $seed" >> $log_file
        echo "file-k: $k" >> $log_file
        echo "file-am: AM$am" >> $log_file
    done
    echo ""
    echo "Add to table..."
    python tablegen.py $log_file >> $output_dir/labeling.table
    echo "Add to intermediate csv..."
    python csvgen.py $log_file >> $output_dir/intermediate.csv
    #echo "Make solution csv..."
    python mk_solution_csv.py $log_file $node_mapping_file > $csv_dir/$data_name.csv
done

### generate real csv from intermediate csv
python mk_final_csv.py $output_dir/intermediate.csv > $output_dir/final.csv

### generate and open table

cat header.labeling $output_dir/labeling.table footer.labeling > $output_dir/labeling.table.tex
cd $output_dir
pdflatex labeling.table.tex
pdflatex labeling.table.tex
open -a Skim labeling.table.pdf
