#!/bin/bash
#SBATCH -J ray_tune
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 96000 # Memory request (in MB)
#SBATCH -t 0-02:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=jbullwinkel@fas.harvard.edu
module load Anaconda3/5.0.1-fasrc01
conda env create -f environment_from_history.yml
source activate denn
cd ../denn
python ray_tune.py --pkey sir