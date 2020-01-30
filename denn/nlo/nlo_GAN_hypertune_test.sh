#!/bin/bash
#SBATCH -J nlo_GAN_hypertune_test
#SBATCH -p test
#SBATCH -n 20
#SBATCH -N 1
#SBATCH --mem 20000 # Memory request (in MB)
#SBATCH -t 0-00:10 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python ../hypertune.py --pkey nlo --gan --nreps 5 --fname nlo_GAN_hypertune_5reps_test_n200.csv
