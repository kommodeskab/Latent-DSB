#!/bin/sh

# SET JOB NAME
#BSUB -J dsb_iterations

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
USE_CLIPPED_LIBRI=True
EXPERIMENT_ID=020625101550

for DSB_ITERATION in 0 1 2 3 4 5 6 7 8
do
  python inference.py --experiment_id=$EXPERIMENT_ID --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --length_seconds=$LENGTH_SECONDS --dsb_iteration=$DSB_ITERATION --noise_factor=0.0 --use_clipped_libri=$USE_CLIPPED_LIBRI --folder_name=dsb_iterations
done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/