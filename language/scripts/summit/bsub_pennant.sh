#!/bin/bash
#BSUB -P CSC275Chen
#BSUB -W 1:00
#BSUB -nnodes 2
#BSUB -alloc_flags smt2
#BSUB -o lsf-%J.out
#BSUB -e lsf-%J.err

root_dir="$PWD"

source "$root_dir"/env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD"

export REALM_BACKTRACE=1

export GASNET_NUM_QPS=1 # Hack: workaround for GASNet bug 3447

for n in 2; do
    for r in 0; do
        if [[ ! -f out_"$n"x2_r"$r".log ]]; then
            echo "Running $n""x2_r$r""..."
            jsrun -n $(( n * 2 )) --tasks_per_rs 1 --rs_per_host 2 --cpu_per_rs 21 --bind rs "$root_dir"/pennant.spmd18 "$root_dir/pennant.tests/leblanc_long4x30/leblanc.pnt" -npieces $(( $n * 18 )) -numpcx 1 -numpcy $(( $n * 18 )) -seq_init 0 -par_init 1 -print_ts 1 -prune 5 -ll:cpu 18 -ll:io 1 -ll:util 2 -ll:dma 2 -ll:csize 60000 -ll:rsize 512 -ll:gsize 0 -hl:prof $(( n * 2 )) -hl:prof_logfile prof_"$n"x2_r"$r"_%.gz -dm:memoize | tee out_"$n"x2_r"$r".log
        fi
    done
done