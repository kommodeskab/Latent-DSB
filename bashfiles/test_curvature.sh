#!/bin/sh

# SET JOB NAME
#BSUB -J curve_test

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 4

# gb memory per core
#BSUB -R "rusage[mem=2G]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 3:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.11.9
source .venv/bin/activate
python inference.py --experiment_name=trajectory_curvature --experiment_id=180425125453 --num_samples=256 --batch_size=128