#!/bin/bash

python3 ./test.py testing start 

# Define the desired --num values in an array
# nums=(60000)
nums=(20000000)

# lac=(5)
mod=(7 8)
number_of_runs=1
# file_error=(22)

current_time=$(date "+%Y%m%d-%H%M%S")
# Define output directories
# output_dir="/mnt/lac-sec/ad-wt-bour/bourbon&wt-last/bourbon/"
output_dir="/mnt/analysis_bourbon/ycsb/"

test_dir="/home/eros/workspace-lsm/wildturkey/build/"

# total_experiment="/mnt/1tb/lac_experiment/"



# Create output directories if they do not exist
if [ ! -d "$output_dir" ]; then
   mkdir -p "$output_dir"
fi

# if [ ! -d "$total_experiment" ]; then
#    mkdir -p "$total_experiment"
# fi


for num in "${nums[@]}"; do

   for md in "${mod[@]}"; do

      for i in $(seq 1 $number_of_runs); do

               output_file="${output_dir}mod=${md}-uni-num=${num}_${i}.txt"
               
               echo "Running db_bench with --num=$num  --mod=${md} " > "$output_file"

               # Run the benchmark
               # uni40,uniread,stats
               # osm_w,real_r,stats
               # fillrandom,readrandom
               # --lac=$lacd 
               # --bwise=$bw
               # --max_file_size=$max
               # --lsize=${max/2} 
               # --file_error=$err
               # f=$((max / 2)) 
               # --lsize=$f
               ${test_dir}/db_bench --benchmarks="uniwrite,ycsba,ycsbb,ycsbc,ycsbd,ycsbe,ycsbf,stats" --mod=$md --num=$num >> "$output_file"
               echo "-------------------------------------" >> "$output_file"

               sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
      done

   done
done


# Execute the db_bench command for each configuration and save results
for num in "${nums[@]}"; do

   for md in "${mod[@]}"; do

      for i in $(seq 1 $number_of_runs); do

         output_file="${output_dir}mod=${md}-zip-1-num=${num}_${i}.txt"
               
         echo "Running db_bench with --num=$num  --mod=${md} " > "$output_file"


         ${test_dir}/db_bench --benchmarks="zipfill,ycsba,ycsbb,ycsbc,ycsbd,ycsbf,stats" --mod=$md --uni=0 --num=$num >> "$output_file"
         echo "-------------------------------------" >> "$output_file"

         sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
      done

   done
done



python3 ./test.py testing end
