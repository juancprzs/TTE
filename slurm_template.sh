#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH --array=[1-25]
#SBATCH -J chunked_baseline
#SBATCH -o logs/chunked_baseline.%J.out
#SBATCH -e logs/chunked_baseline.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=perezjc@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A conf-gpu-2020.11.23

python main.py \
--checkpoint runs/remove_soon \
--experiment local_trades
