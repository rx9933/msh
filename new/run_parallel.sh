#!/bin/bash

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Function to print timestamped messages
log_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
PROJECT_NAME='pml' # or free_surface
log_time "Starting parallel data generation..." 
nsamplesperjob=1
njobs=1
# for ((seed=0; seed<=njobs; seed++)); do
#     log_time "Launching job with seed ${seed}"
#     # python generate_data.py --n_data ${nsamplesperjob} --seed ${seed} --job_id ${seed} >& log${seed}.txt &
#     python generate_data_posa.py --j_rank 1000 --n_data ${nsamplesperjob} --seed ${seed} --job_id ${seed} >& log${seed}.txt &
#     pid=$!
#     log_time "Job ${seed} started with PID ${pid}"
# done

log_time "now regular data not posa gen..."
for ((seed=0; seed<=njobs; seed++)); do
    log_time "Launching job with seed ${seed}"
    # python generate_data.py --bc ${PROJECT_NAME} --n_data ${nsamplesperjob} --seed ${seed} --job_id ${seed} >& log${seed}.txt &
    python generate_data_posa.py --bc ${PROJECT_NAME} --j_rank 200 --n_data ${nsamplesperjob} --seed ${seed} --job_id ${seed} >& log${seed}.txt &
    pid=$!
    log_time "Job ${seed} started with PID ${pid}"
done

log_time "Waiting for all jobs to complete..."
wait
log_time "All jobs finished"

# for seed in {0..5}; do
#     echo "Starting job with seed ${seed}"
#     python generate_data.py --seed ${seed} --job_id ${seed} >& log${seed}.txt &
#     echo "Job ${seed} started with PID $!"
# done;
# wait
# echo "All jobs completed"


#!/bin/bash
# python3 generate_data.py --seed 0 --job_id 0 &
# python3 generate_data.py --seed 1 --job_id 1 &
# python3 generate_data.py --seed 2 --job_id 2 &
# wait # Optional: waits for all background processes to complete