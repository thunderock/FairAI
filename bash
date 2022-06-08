#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=blazor
#SBATCH --cpus-per-task=20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ashutiwa@iu.edu


srun python driver.py
