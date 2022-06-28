#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=FairAI_slate
#SBATCH --cpus-per-task=100
#SBATCH --mail-type=ALL
#SBATCH --mem=240G
#SBATCH --mail-user=ashutiwa@iu.edu
#SBATCH --partition=general


srun python driver_torch.py
