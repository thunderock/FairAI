#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=blazer
#SBATCH --cpus-per-task=100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashutiwa@iu.edu
#SBATCH --partition=general


srun python driver.py
