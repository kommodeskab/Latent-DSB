from src.lightning_modules.dsb import load_dsb_model, load_dsb_datasets
from src.dataset import ClippedLibri, EarsWHAMUnpaired
import torch
from torch import Tensor
from argparse import ArgumentParser
from src.callbacks.utils import get_batch_from_dataset

torch.manual_seed(0)
# make an argument parser for experiment_name
parser = ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='noise_variance')
parser.add_argument('--experiment_id', type=str, default='180425125453')
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=128)

args = parser.parse_args()
experiment_name = args.experiment_name
experiment_id = args.experiment_id
num_samples = args.num_samples
batch_size = args.batch_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if experiment_name == 'noise_variance':
    dsb_iteration = 10
    
    dsb = load_dsb_model(experiment_id, dsb_iteration)
    x0_dataset, x1_dataset = load_dsb_datasets(experiment_id, dsb_iteration)
    
    if experiment_id == '180425125453':
        dsb.encoder_decoder.off_set = 0
    
    x1_dataset : EarsWHAMUnpaired
    snr_levels = torch.linspace(-2, 18, 10, dtype=torch.int16).tolist()
    batches : list[Tensor] = []
    for snr in snr_levels:
        sample = x1_dataset.get_item(0, snr)[1]
        batch = sample.repeat(num_samples, 1, 1)
        batches.append(batch)
    
    data_dict = {}
    
    for snr, batch in zip(snr_levels, batches):
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            batch_encoded = dsb.encode(batch)
            x0_hat_encoded = dsb.chunk_sample(batch_encoded, forward=False, chunk_size=batch_size, show_progress=True, noise='inference')
            
        data_dict[snr] = x0_hat_encoded.cpu()
    
    torch.save(data_dict, f"test_results/noise_variance.pt")

elif experiment_name == 'trajectory_curvature':
    import os
    dsb_iterations = []
    for i in range(1, 15):
        if os.path.exists(f'logs/dsb/{experiment_id}/checkpoints/DSB_iteration_{i}.ckpt'):
            dsb_iterations.append(i)
            
    print(f"Found DSB iterations:", ", ".join([str(i) for i in dsb_iterations]))
    
    data_dict = dict()
    x0_dataset, x1_dataset = load_dsb_datasets(experiment_id, 1)
    x1_dataset : EarsWHAMUnpaired | ClippedLibri
    x1_batch = get_batch_from_dataset(x1_dataset, num_samples, shuffle=True)
    x1_batch = x1_batch.to(device)
    
    for dsb_iteration in dsb_iterations:
        dsb = load_dsb_model(experiment_id, dsb_iteration)
        
        if experiment_id == '180425125453':
            dsb.encoder_decoder.off_set = 0
        
        if dsb_iteration == dsb_iterations[0]:
            timeschedule = dsb.scheduler.gammas_bar
            data_dict['timeschedule'] = timeschedule.cpu()
            
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            x1_encoded = dsb.encode(x1_batch)
            x0_hat_encoded = dsb.chunk_sample(x1_encoded, forward=False, chunk_size=batch_size, return_trajectory=True, show_progress=True, noise='none')
            
        data_dict[dsb_iteration] = x0_hat_encoded.cpu()
        
    torch.save(data_dict, f"test_results/trajectory_curvature.pt")