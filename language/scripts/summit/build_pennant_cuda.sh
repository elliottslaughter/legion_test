#!/bin/bash

set -e

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

mkdir "$1"
cd "$1"

for c in 3; do
    SAVEOBJ=1 STANDALONE=1 OBJNAME=./pennant.spmd"$c" $root_dir/../regent.py $root_dir/../examples/pennant_fast.rg -fflow 1 -fflow-spmd 1 -fflow-spmd-shardsize "$c" -fvectorize-unsafe 1 -fopenmp 0 -fcuda 1 -fcuda-offline 1
done

cp -r $root_dir/../examples/pennant.tests .

cp $root_dir/../scripts/*_pennant*.sh .
cp $root_dir/../scripts/summit_env.sh .
cp $root_dir/../scripts/summarize.py .