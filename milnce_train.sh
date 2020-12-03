#!/bin/bash

# export SEED_LST=$"42 68 99"
export OUT_HEAD="job_outputs/out-"
export ERR_HEAD="job_outputs/err-"
export SEED_LST="42 64"
export CAND_LST="1 4 7"
# export NUM_NODE=1
export NUM_GRU_NODE=2

for CANDIDATE in $CAND_LST;
  do
  for SEED in $SEED_LST;
    do
    echo "______________________________"
    echo "Executing MILNCE for seed=$SEED & candidate=$CANDIDATE"
    name="milnce-seed_$SEED-candidate_$CANDIDATE"
    # OUT_FILE="$OUT_HEAD$name.txt"
    # ERR_FILE="$ERR_HEAD$name.txt"
    bash milnce_train_single.sh $SEED $CANDIDATE $name
    # sbatch -J $name -o $OUT_FILE  -e $ERR_FILE -t 4:00:00 -p gpu --gres=gpu:$NUM_GRU_NODE --mem=16G train_single.sh $SEED $CANDIDATE
    echo "Done."
    done
  done
