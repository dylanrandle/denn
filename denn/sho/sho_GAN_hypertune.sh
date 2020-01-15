#!/bin/bash
#SBATCH -J sho_GAN_hypertune
#SBATCH -p shared
#SBATCH -n 500
#SBATCH --mem 100000 # Memory request (in MB)
#SBATCH -t 0-48:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python hypertune.py --ncpu 500 --fname sho_GAN_hypertune_biggest_ever.csv --gan
