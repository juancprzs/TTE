#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[0-2]
#SBATCH -J crops2
#SBATCH -o logs/crops2.%J.out
#SBATCH -e logs/crops2.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL

python main.py --seed ${SLURM_ARRAY_TASK_ID} \
--checkpoint runs/crops2_run${SLURM_ARRAY_TASK_ID} \
--n-crops 2
