#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-2]
#SBATCH -J flip_crops4
#SBATCH -o logs/flip_crops4.%J.out
#SBATCH -e logs/flip_crops4.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23

python main.py --seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/flip_crops4_run${SLURM_ARRAY_TASK_ID} \
--n-crops 4 \
--flip \
--flip-crop
