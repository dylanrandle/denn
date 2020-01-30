#!/bin/bash
#SBATCH -J sho_GAN_niters_test
#SBATCH -p test
#SBATCH -n 20
#SBATCH --mem 4000 # Memory request (in MB)
#SBATCH -t 0-01:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python ../niters.py --pkey sho --gan --nreps 20 --fname sho_GAN_niters_test_generalized.csv
