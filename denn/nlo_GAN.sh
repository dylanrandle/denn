#!/bin/bash
#SBATCH -J nlo_gan
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem 2000 # Memory request (in MB)
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --pkey nlo --gan --fname gan_nlo_test_64by18.png
