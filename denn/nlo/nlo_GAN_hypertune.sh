#!/bin/bash
#SBATCH -J nlo_GAN_hypertune
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 26000 # Memory request (in MB)
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python hypertune.py --gan --fname nlo_GAN_hypertune_10k_hugeD.csv
