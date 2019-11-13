#!/bin/bash
#SBATCH -J obs_every_experiment
#SBATCH -p shared
#SBATCH -n 12
#SBATCH -N 1
#SBATCH --mem 10000 # Memory request (in MB)
#SBATCH -t 0-08:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
module load Anaconda3/5.0.1-fasrc01
source activate denn
python obs_every_experiment.py
