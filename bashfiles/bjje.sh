#!/bin/sh

# SET JOB NAME
#BSUB -J noise_mel

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q p1

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=2GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 71:59
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.12
source .venv/bin/activate
python3 train.py +experiment=fr_init_noise_mel data.batch_size=16