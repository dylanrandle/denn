#!/bin/bash
#SBATCH -J sho_L2_hypertune
#SBATCH -p test
#SBATCH -n 20
#SBATCH --mem 2000 # Memory request (in MB)
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python hypertune.py --ncpu 1 --fname sho_L2_niters_20rep_60by2.csv
