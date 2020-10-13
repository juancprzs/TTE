#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-2]
#SBATCH -J sep_flip_crops2
#SBATCH -o logs/sep_flip_crops2.%J.out
#SBATCH -e logs/sep_flip_crops2.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python main.py --test-samples 2500 --seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/sample_sep_flip_crops2_run${SLURM_ARRAY_TASK_ID} \
--n-crops 2 \
--flip
