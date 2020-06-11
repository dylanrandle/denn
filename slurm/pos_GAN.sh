#!/bin/bash
#SBATCH -J pos_gan
#SBATCH -p test
#SBATCH -n 1
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-00:10 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python experiments.py --gan --pkey pos --fname pos_gan_1kiter.png
