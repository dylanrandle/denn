#!/bin/bash
#SBATCH -J nlo_gan
#SBATCH -p test
#SBATCH -n 32
#SBATCH -N 1
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --pkey nlo --gan --fname gan_nlo_L2_64by12.png
