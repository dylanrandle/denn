#!/bin/bash
#SBATCH -J sho_gan
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem 200 # Memory request (in MB)
#SBATCH -t 0-00:10 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --gan --pkey sho --fname sho_gan_test.png
