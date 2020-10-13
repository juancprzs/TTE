#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-1]
#SBATCH -J flip
#SBATCH -o logs/flip.%J.out
#SBATCH -e logs/flip.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python main.py --seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/flip_run${SLURM_ARRAY_TASK_ID} \
--flip
