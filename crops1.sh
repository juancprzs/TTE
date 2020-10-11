#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-2]
#SBATCH -J crops1
#SBATCH -o logs/crops1.%J.out
#SBATCH -e logs/crops1.%J.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23


python main.py \
--seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/n_crops1_run${SLURM_ARRAY_TASK_ID} \
--n-crops 1
