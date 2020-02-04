#!/bin/bash
#SBATCH -J nlo_L2_niters_test
#SBATCH -p shared
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-00:30 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python ../niters.py --pkey nlo --nreps 20 --fname nlo_L2_niters_test_generalized.csv
