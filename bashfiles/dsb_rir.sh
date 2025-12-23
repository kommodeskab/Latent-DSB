#!/bin/sh

# SET JOB NAME
#BSUB -J rir

# select gpu, choose gpuv100, gpua100 or p1 (h100)
#BSUB -q gpua100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=2G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 48:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.11.9
source .venv/bin/activate
python3 train.py +experiment=dsb_rir_stft trainer.log_every_n_steps=400