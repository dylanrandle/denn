#!/bin/bash
#SBATCH -J nlo_GAN_hypertune
#SBATCH -p shared
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-16:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python hypertune.py --gan --ncpu 32 --fname nlo_GAN_hypertune.csv
