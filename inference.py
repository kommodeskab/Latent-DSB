from src.lightning_modules.dsb import load_dsb_model, load_dsb_datasets, get_dsb_iterations
from src.dataset import ClippedLibri, EarsWHAMUnpaired
import torch
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

length_seconds = 2.23

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if experiment_name == 'noise_variance':
    dsb_iteration = max(get_dsb_iterations(experiment_id))
    print(f"Found latest DSB iteration:", dsb_iteration)
    
    dsb = load_dsb_model(experiment_id, dsb_iteration)
    x0_dataset, x1_dataset = load_dsb_datasets(experiment_id)
    x1_dataset : EarsWHAMUnpaired | ClippedLibri
    
    x1_dataset.set_length(length_seconds)
    x1_dataset.return_pair = False
    min_noise, max_noise = x1_dataset.noise_range
    noise_levels = torch.linspace(min_noise, max_noise, 10, dtype=torch.int16).tolist()
    print(f"Noise levels: {' '.join([str(i) for i in noise_levels])}")
    data_dict = dict()
    
    for noise_level in noise_levels:
        print("Current noise level:", noise_level)
        noisy_sample = x1_dataset.get_item(0, noise_level)
        batch = noisy_sample.repeat(num_samples, 1, 1)

        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            batch_encoded = dsb.encode(batch)
            x0_hat = dsb.chunk_sample(batch_encoded, forward=False, chunk_size=batch_size, show_progress=True, noise='inference')
            # x0_hat.shape = (batch_size, ...)
            x0_hat_flat = x0_hat.cpu().reshape(x0_hat.shape[0], -1)
            avg_stdev = x0_hat_flat.std(dim=0).mean()
            
            if noise_level == noise_levels[0]:
                sanity_check_sample = dsb.decode(x0_hat)[0]
                data_dict['sanity_check_sample'] = sanity_check_sample.cpu()
            
        data_dict[noise_level] = avg_stdev.item()
    
    torch.save(data_dict, f"test_results/noise_variance_{experiment_id}.pt")

elif experiment_name == 'trajectory_curvature':
    from src.callbacks.metrics import calculate_curvature_displacement
    
    dsb_iterations = get_dsb_iterations(experiment_id)
    print(f"Found DSB iterations:", ", ".join([str(i) for i in dsb_iterations]))
    
    data_dict = dict()
    _, x1_dataset = load_dsb_datasets(experiment_id)
    x1_dataset : EarsWHAMUnpaired | ClippedLibri
    
    x1_dataset = x1_dataset.set_length(length_seconds)
    x1_dataset.return_pair = False
    x1_batch = get_batch_from_dataset(x1_dataset, num_samples, shuffle=True)
    x1_batch = x1_batch.to(device)
    
    for dsb_iteration in dsb_iterations:
        print("Current DSB iteration:", dsb_iteration)
        dsb = load_dsb_model(experiment_id, dsb_iteration)
        
        if dsb_iteration == dsb_iterations[0]:
            timeschedule = dsb.scheduler.gammas_bar
            data_dict['timeschedule'] = timeschedule
            
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            x1_encoded = dsb.encode(x1_batch)
            trajectory = dsb.chunk_sample(x1_encoded, forward=False, chunk_size=batch_size, return_trajectory=True, show_progress=True, noise='none').cpu()
            C_t = calculate_curvature_displacement(trajectory, timeschedule)
            
        data_dict[dsb_iteration] = C_t
        
    torch.save(data_dict, f"test_results/trajectory_curvature_{experiment_id}.pt")