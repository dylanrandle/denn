#!/bin/bash
#SBATCH -J nlo_L2
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 20000 # Memory request (in MB)
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --pkey nlo --fname nlo_L2_best_200k.png
