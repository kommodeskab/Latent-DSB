#!/bin/sh

# SET JOB NAME
#BSUB -J noise_factor

# select gpu, choose gpuv100 or gpua100 (best)
#BSUB -q gpuv100

# number of GPUs to use
#BSUB -gpu "num=1:mode=exclusive_process"

# number of cores to use
#BSUB -n 12

# gb memory per core
#BSUB -R "rusage[mem=2GB]"
# cores is on the same slot
#BSUB -R "span[hosts=1]"

# walltime
#BSUB -W 4:00
#BSUB -o hpc/output_%J.out 
#BSUB -e hpc/error_%J.err 

module load python3/3.11.9
source .venv/bin/activate

LENGTH_SECONDS=4.47
NUM_SAMPLES=256
BATCH_SIZE=64
EXPERIMENT_ID=020625101550
USE_CLIPPED_LIBRI=True

for NOISE_FACTOR in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0
do
  python inference.py --experiment_id=$EXPERIMENT_ID --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --length_seconds=$LENGTH_SECONDS --noise_factor=$NOISE_FACTOR --use_clipped_libri=$USE_CLIPPED_LIBRI --folder_name=noise_factor
done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/