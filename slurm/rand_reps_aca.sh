#!/bin/bash
#SBATCH -J rand_reps_aca
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
# python rand_reps.py --pkey aca --nreps 10 --fname aca_rand_reps_gan
python rand_reps.py --pkey aca --loss MSELoss --nreps 10 --fname aca_rand_reps_L2
python rand_reps.py --pkey aca --loss L1Loss --nreps 10 --fname aca_rand_reps_L1
python rand_reps.py --pkey aca --loss SmoothL1Loss --nreps 10 --fname aca_rand_reps_huber