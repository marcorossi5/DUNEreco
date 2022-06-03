#!/bin/bash
# Usage (from DUNEdn root folder): benchmarks/make_inference.sh

model=$1 # one of uscg | cnn | gcnn
version=$2 # one of v08 | v09
extra_flags=$3
fname="$model"_"$version"
outdir=benchmarks/onnx/onnx_benchmark

# mkdir -p $outdir
# echo Extracting data into $outdir...
# tar -xf examples/dunetpc_inspired_v09_p2GeV_rawdigits.tar.gz -C $outdir
python benchmarks/compute_denoising_performance.py \
    -i $outdir/p2GeV_cosmics_inspired_rawdigit_evt8.npy \
    -o $outdir/p2GeV_cosmics_inspired_rawdigit_denoised_"$fname"_evt8.npy\
    -m $model \
    -t $outdir/p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy \
    --model_path ../saved_models/$fname \
    $extra_flags