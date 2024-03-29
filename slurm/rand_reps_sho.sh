#!/bin/bash
#SBATCH -J rand_reps_sho
#SBATCH -p test
#SBATCH -n 48
#SBATCH -N 1
#SBATCH --mem 96000 # Memory request (in MB)
#SBATCH -t 0-04:00 # Maximum execution time (D-HH:MM)
#SBATCH -o logs.out # Standard output
#SBATCH -e logs.err # Standard error
#SBATCH --mail-type=END
#SBATCH --mail-user=jbullwinkel@fas.harvard.edu
module load gcc/10.2.0-fasrc01
module load Anaconda3/2020.11
source activate denn
cd ../denn
python rand_reps.py --pkey sho --fname sho_rand_reps_huber
#python rand_reps.py --pkey sho --gan --nreps 20 --fname rand_reps_sho