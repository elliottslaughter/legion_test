#!/bin/bash
#BSUB -P CSC275Chen
#BSUB -W 1:00
#BSUB -nnodes 1
#BSUB -o lsf-%J.out
#BSUB -e lsf-%J.err

root_dir="$PWD"

source "$root_dir"/summit_env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"

export REALM_BACKTRACE=1

for n in 1; do
    for r in 0; do
        if [[ ! -f out_"$n"x2_r"$r".log ]]; then
            echo "Running $n""x2_r$r""..."
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --gpu_per_rs 3 --bind rs --smpiargs="-x PAMI_DISABLE_CUDA_HOOK=1 -disable_gpu_hooks" "$root_dir"/taylor-312x156x156-dop"$n" -ll:cpu 2 -ll:gpu 3 -ll:fsize 8192 -ll:show_rsrv -hl:prof $(( n * 2 )) -hl:prof_logfile prof_"$n"x2_r"$r".gz | tee out_"$n"x2_r"$r".log
        fi
    done
done