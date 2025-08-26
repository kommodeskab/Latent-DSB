#!/bin/sh

# SET JOB NAME
#BSUB -J test_traj

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
#BSUB -W 3:00
#BSUB -o hpc/test_traj.out 
#BSUB -e hpc/test_traj.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

# python inference.py --experiment_id=GFB                 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --save_trajectory=True --num_steps=50                    --folder_name=test_traj/gfb  --what_test=clip
python inference.py --experiment_id=ESDSB_230825112100  --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --save_trajectory=True --num_steps=50 --noise_factor=0.0 --folder_name=test_traj/stft  --what_test=clip
python inference.py --experiment_id=ESDSB_120825012451  --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --save_trajectory=True --num_steps=50 --noise_factor=0.0 --folder_name=test_traj/mel   --what_test=clip

chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/