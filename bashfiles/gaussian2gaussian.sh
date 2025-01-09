#!/bin/sh

# SET JOB NAME
#BSUB -J g2g

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 4

# gb memory per core
#BSUB -R "rusage[mem=4G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 4:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.12.4
source .venv/bin/activate
python train.py +experiment=gaussian_test dim=16
python train.py +experiment=gaussian_test dim=32
python train.py +experiment=gaussian_test dim=64
python train.py +experiment=gaussian_test dim=256