#!/bin/bash
#SBATCH -J pos_GAN_hypertune_test
#SBATCH -p test
#SBATCH -n 20
#SBATCH --mem 20000 # Memory request (in MB)
#SBATCH -t 0-00:30 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python ../hypertune.py --pkey pos --gan --nreps 1 --fname pos_GAN_hypertune_test_generalized.csv
