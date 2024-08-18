#!/bin/sh
#SBATCH --time=0-6
#SBATCH --array=1-143
#SBATCH --gres=gpu:A4000:1
#SBATCH --mem=64G
#SBATCH --requeue

task=$(sed -n "$SLURM_ARRAY_TASK_ID"p tasks.txt)
python3 main.py --task $task --config "no"
python3 main.py --task $task --config "replace"
python3 main.py --task $task --config "concat"
python3 main.py --task $task --config "mean"