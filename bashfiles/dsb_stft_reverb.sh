#!/bin/sh

# SET JOB NAME
#BSUB -J reverb_stft

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q p1

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=6GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 71:59
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err   

module load python3/3.11.9
source .venv/bin/activate
python3 train.py +experiment=dsb_stft_reverb model.num_workers=12
chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/