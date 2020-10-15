#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-10]
#SBATCH -J dist_flip
#SBATCH -o logs/dist_flip.%J.out
#SBATCH -e logs/dist_flip.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23

python main.py \
--checkpoint runs/distributed_flip \
--num-chunk ${SLURM_ARRAY_TASK_ID} \
--flip