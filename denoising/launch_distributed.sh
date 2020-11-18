#!/usr/bin/bash

### Example command to run: >>> ./launch.sh "3 3 3"###

# This script launches the distributed training from the local node
# It is possible to choose beetwen different setups, editing config section
# The user must input the number of gpus he wants to use for each node:
# (# ibmminsky-1, # ibmmisnky-2, # ibmminsky-3)
# Note: each node must have the same number of spawned processes, equal or less
# than the # of GPUs on the node, otherwise everything will break !
# The master node will be automatically the last nonzero node with GPU usage
# It is mandatory to launch the job from the master node

# General comment about bash
# ${} parameter expansion
# $() command substitution
# <() or >() process substitution
# ${EXPR} before $(command) before <(process)

# config
email="marco.rossi@cern.ch"
workdir="/nfs/public/romarco/DUNEreco"
directory="${workdir}/denoising"
logdir="${directory}/logdir"

setenv="source /afs/cern.ch/user/r/romarco/setup_wmla"
$setenv
launch=$(python -c "import torch.distributed.launch as t; print(t.__file__)")
main=${directory}/denoise.py

minsky_IPs=(128.142.165.77 \
            128.142.165.78 \
            128.142.165.79)

gpus=($1)
card=$2

# filter unused nodes

num_gpus=0
for idx in ${!gpus[*]}; do
    if [ ${gpus[$idx]} -eq 0 ]; then
        unset gpus[$idx]
    fi
done

nnodes=${#gpus[*]}

if [ $nnodes -eq 0 ]; then
    echo "Error: Number of GPUs must be greater than zero"
    exit -1
else
    echo "Running on $(($nnodes * ${gpus[-1]})) GPUs"
fi

hosts=(${!gpus[*]})
master_addr=${minsky_IPs[${hosts[-1]}]}
#add 1 to hosts
for idx in ${!hosts[*]}; do
    hosts[$idx]=$((${hosts[$idx]}+1))
done

function job_func(){
    job="python $launch --nproc_per_node=$2 --nnodes=$nnodes --node_rank=$1 \
         --master_addr=$master_addr $main --local_world_size=$2 --card=$card"
}
trap 'rm -f "$logdir"/tmp*' EXIT # automatic clean of tmp files
separator="\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n"

rank=$(( $nnodes - 1 ))
rank_h=0
for gpu in ${gpus[*]}; do
    host=${hosts[$rank_h]}
    job_func $rank $gpu
    if [ $rank -gt 0 ]; then
        job="$setenv;cd $workdir;$job"
        nohup ssh -K ibmminsky-$host $job &
    else
        $job
        returncode=$?
    fi
    rank=$(( $rank - 1 ))
    rank_h=$(( $rank_h + 1 ))
done

# send logs to email
if [[ 0 -ne "$returncode" ]]; then
    echo FAIL
else
    echo SUCCESS;
fi
