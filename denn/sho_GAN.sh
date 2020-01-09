#!/bin/bash
#SBATCH -J sho_gan
#SBATCH -p test
#SBATCH -n 12
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --gan --pkey sho --fname sho_gan_100k.png
