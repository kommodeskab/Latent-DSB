#!/bin/sh

# SET JOB NAME
#BSUB -J test_noise

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
#BSUB -W 12:00
#BSUB -o hpc/test_noise.out 
#BSUB -e hpc/test_noise.err 

module load python3/3.11.9
source .venv/bin/activate

NUM_SAMPLES=256
BATCH_SIZE=16

python inference.py --experiment_id=baseline --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1 --folder_name=test_noise/baseline --what_test=noise
python inference.py --experiment_id=sepformer --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1 --folder_name=test_noise/sepformer --what_test=noise 
python inference.py --experiment_id=convtasnet --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1 --folder_name=test_noise/convtasnet --what_test=noise 

# stft
python inference.py --experiment_id=ESDSB_090925111017 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=50 --noise_factor=1.0 --folder_name=test_noise/stft_sto_50 --what_test=noise 
python inference.py --experiment_id=ESDSB_090925111017 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1  --noise_factor=1.0 --folder_name=test_noise/stft_sto_1 --what_test=noise 

# mel
python inference.py --experiment_id=ESDSB_130825172731 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=1 --noise_factor=1.0 --folder_name=test_noise/esdsb_sto_1 --what_test=noise 
python inference.py --experiment_id=ESDSB_130825172731 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=50 --noise_factor=1.0 --folder_name=test_noise/esdsb_sto_50 --what_test=noise 
python inference.py --experiment_id=ESDSB_130825172731 --num_samples=$NUM_SAMPLES --batch_size=$BATCH_SIZE --num_steps=50 --noise_factor=0.0 --folder_name=test_noise/esdsb_det_50 --what_test=noise 


chgrp -R s214630bjjemiri /work3/s214630/Latent-DSB/
chmod -R 770 /work3/s214630/Latent-DSB/