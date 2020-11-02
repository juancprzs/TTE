#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-32]
#SBATCH -J abl_eps
#SBATCH -o logs/abl_eps.%J.out
#SBATCH -e logs/abl_eps.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23

python main.py \
--checkpoint runs/obf_abl_${SLURM_ARRAY_TASK_ID}of255 \
--eps ${SLURM_ARRAY_TASK_ID} \
--experiment local_trades
