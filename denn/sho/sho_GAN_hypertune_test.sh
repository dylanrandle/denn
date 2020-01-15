#!/bin/bash
#SBATCH -J sho_GAN_hypertune_test
#SBATCH -p shared
#SBATCH -n 96
#SBATCH --mem 4000 # Memory request (in MB)
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python hypertune.py --ncpu 96 --fname sho_GAN_hypertune_test4.csv --gan
