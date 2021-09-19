#!/bin/bash
#SBATCH -J ray_tune_nlo
#SBATCH -p shared
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 96000 # Memory request (in MB)
#SBATCH -t 0-05:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
#SBATCH --mail-type=END
#SBATCH --mail-user=jbullwinkel@fas.harvard.edu
module load gcc/10.2.0-fasrc01
module load Anaconda3/2020.11
source activate denn
cd ../denn
python ray_tune.py --pkey nlo --ncpu 48 --nsample 400