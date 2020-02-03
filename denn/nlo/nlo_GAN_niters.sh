#!/bin/bash
#SBATCH -J nlo_GAN_niters
#SBATCH -p shared
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --mem 14000 # Memory request (in MB)
#SBATCH -t 0-60:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python ../niters.py --pkey nlo --gan --nreps 20 --fname nlo_GAN_niters_new_to_200k.csv
