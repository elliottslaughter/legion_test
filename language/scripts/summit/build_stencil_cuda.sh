#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

export USE_FOREIGN=0

for c in 3; do
    SAVEOBJ=1 STANDALONE=1 OBJNAME=./stencil.spmd"$c" $root_dir/../../regent.py $root_dir/../../examples/stencil_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fopenmp 0 -fcuda 1 -fcuda-offline 1
done

cp $root_dir/*_stencil_cuda*.sh .
cp $root_dir/../summarize.py .

cp $root_dir/env.sh .