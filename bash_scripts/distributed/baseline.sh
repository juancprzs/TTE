#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-10]
#SBATCH -J baseline
#SBATCH -o logs/baseline.%J.out
#SBATCH -e logs/baseline.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python main.py \
--checkpoint runs/distributed_baseline \
--num-chunk ${SLURM_ARRAY_TASK_ID}