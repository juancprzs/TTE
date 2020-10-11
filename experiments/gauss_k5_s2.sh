#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-1]
#SBATCH -J gauss_k5_s2
#SBATCH -o logs/gauss_k5_s2.%J.out
#SBATCH -e logs/gauss_k5_s2.%J.err
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python main.py --test-samples 2500 --seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/sample_gauss_k5_s2_run${SLURM_ARRAY_TASK_ID} \
--gauss-k 5 \
--gauss-s 2
