#!/bin/bash
#SBATCH -J ray_tune_burv
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 96000 # Memory request (in MB)
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
#SBATCH --mail-type=END
#SBATCH --mail-user=jbullwinkel@fas.harvard.edu
module load gcc/10.2.0-fasrc01
module load Anaconda3/2020.11
source activate denn
cd ../denn
python ray_tune.py --pkey burv --loss MSELoss --ncpu 48 --nsample 200
python ray_tune.py --pkey burv --loss L1Loss --ncpu 48 --nsample 200
python ray_tune.py --pkey burv --loss SmoothL1Loss --ncpu 48 --nsample 200
# python ray_tune.py --pkey burv --ncpu 48 --nsample 1500
