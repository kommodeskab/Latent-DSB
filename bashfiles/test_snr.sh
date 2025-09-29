#!/bin/sh

# SET JOB NAME
#BSUB -J test_snr

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
#BSUB -o hpc/test_snr.out 
#BSUB -e hpc/test_snr.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

for SNR in 0 5 10 15
do
  python inference.py --experiment_id=dsb_130825172731 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=30 --snr=$SNR --folder_name=test_snr/dsb_sto_$SNR --what_test=noise 
done

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/