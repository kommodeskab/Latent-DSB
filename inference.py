import torch
from argparse import ArgumentParser
from src.lightning_modules.dsb import get_dsb_iterations, load_dsb_model
from GFB import GFB
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.dataset import ClippedLibri, LibriRIR
from src.callbacks.metrics import WER, SRCS, DNSMOS, KAD
import random
import numpy as np
from torch import Tensor
from src.lightning_modules import DSB

gfb_schedule_type = 'cosine'
sample_rate = 16000
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

parser = ArgumentParser()
parser.add_argument('--experiment_id', type=str)
parser.add_argument('--num_samples', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_steps', type=int, default=None)
parser.add_argument('--noise_factor', type=float, default=1.0)
parser.add_argument('--dsb_iteration', type=int, default=None)
parser.add_argument('--what_test', type=str, required=True) # clip or rir
parser.add_argument('--folder_name', type=str, required=True)

args = parser.parse_args()
params = {}
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
    params[arg] = value
    
experiment_id : str = args.experiment_id
num_samples : int = args.num_samples
batch_size : int = args.batch_size
num_steps : int = args.num_steps
noise_factor : float = args.noise_factor
dsb_iteration : int = args.dsb_iteration
what_test : str = args.what_test
folder_name : str = args.folder_name

assert what_test in ['rir', 'clip'], "what_test must be either 'rir' or 'clip'"

folder_name = f"test_results/{folder_name}"
os.makedirs(folder_name, exist_ok=True)

if experiment_id == 'GFB':
    assert dsb_iteration is None, "DSB iteration is not supported for GFB"
    assert num_steps > 1, "GFB cant do one-step inference"
    
    length_seconds = 4.096
    
    dsb = GFB(device, what_test=what_test, schedule_type=gfb_schedule_type)
    if num_steps is not None:
        print(f"Setting num_steps to {num_steps}")
        # Tsteps is the number of steps for encoding / decoding, total num_steps = 2 * Tsteps
        dsb.Tsteps = num_steps // 2
    
    timeschedule = dsb.diff.get_schedule(dsb.Tsteps, end_t=1.0, type=gfb_schedule_type)
    
elif experiment_id == 'baseline':
    assert dsb_iteration is None, "DSB iteration is not supported for baseline"
    print("Using baseline model")
    
    length_seconds = 4.096
    class BaselineModel:
        def __init__(self, num_steps : int): 
            self.num_steps = num_steps
        def encode(self, x): return x
        def decode(self, x): return x
        def sample(self, x : torch.Tensor, **kwargs):
            if kwargs.get('return_trajectory', False):
                trajectory = [x] * (self.num_steps + 1)
                trajectory = torch.stack(trajectory, dim=0)
                return trajectory
            return x
        def to(self, device): return self
        
    dsb = BaselineModel(num_steps=num_steps)
    timeschedule = None

else:
    if dsb_iteration is None:
        dsb_iterations = get_dsb_iterations(experiment_id)
        dsb_iteration = max(dsb_iterations)
        params['dsb_iteration'] = dsb_iteration
        print(f"Using DSB model with iteration: {dsb_iteration}")
        
    dsb = load_dsb_model(experiment_id, dsb_iteration)
    dsb.eval()
    if num_steps is not None:
        print(f"Setting num_steps to {num_steps}")
        dsb.scheduler.set_timesteps(num_steps)
    
    from src.networks import STFTEncoderDecoder, HifiGan
    if isinstance(dsb.encoder_decoder, STFTEncoderDecoder):
        length_seconds = 4.096
    elif isinstance(dsb.encoder_decoder, HifiGan):
        length_seconds = 4.47
    else:
        raise ValueError(f"Unknown encoder_decoder type: {type(dsb.encoder_decoder)}")
        
    timeschedule = dsb.scheduler.gammas_bar

dsb.to(device)

if what_test == 'clip':
    dataset = ClippedLibri(length_seconds=5.0, sample_rate=sample_rate, train=False, return_pair=True)
    gain_db = dataset.what_db_for_sdr(target_snr=2.0)
    dataset = [dataset.get_item(i, gain_db=gain_db) for i in range(num_samples)]
elif what_test == 'rir':
    dataset = LibriRIR(length_seconds=5.0, sample_rate=sample_rate, train=False, return_pair=True)
    dataset = [dataset.get_item(i) for i in range(num_samples)]
else:
    raise ValueError(f"Unknown test type: {what_test}")
    
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

wer = WER(sample_rate, device)
srcs = SRCS(sample_rate, device)
dnsmos = DNSMOS(sample_rate, device)
kad = KAD(sample_rate, device)

def encoding_to_waveform(trajectory : Tensor, dsb : DSB) -> Tensor:
    # convert the latent trajectory to waveform trajectory
    traj_len, batch_size, *sample_shape = trajectory.shape
    trajectory = trajectory.reshape(traj_len * batch_size, *sample_shape)
    waveform = dsb.chunk_decode(trajectory, chunk_size=32)
    waveform_shape = waveform.shape[1:]
    waveform = waveform.reshape(traj_len, batch_size, *waveform_shape)
    return waveform
    

for i, batch in enumerate(tqdm(dataloader, desc="Loading batches")):
    x0, x1 = batch
    x0 : torch.Tensor
    x1 : torch.Tensor
    
    seq_len = int(length_seconds * sample_rate)
    x0, x1 = x0.to(device), x1.to(device)
    x0 = x0[:, :, :seq_len]
    x1 = x1[:, :, :seq_len]
    
    x1_encoded = dsb.encode(x1)
    trajectory = dsb.sample(x1_encoded, forward=False, return_trajectory=True, noise='inference', noise_factor=noise_factor, show_progress=True)
    x0_recon_encoded = trajectory[-1]
    x0_recon = dsb.decode(x0_recon_encoded)
    
    if isinstance(dsb, DSB):
        trajectory = encoding_to_waveform(trajectory, dsb)
    
    # squeeze and make sure that length is no longer than 2**16 for fair comparison
    x0_recon = x0_recon.squeeze(1)[:, :2**16]
    x0 = x0.squeeze(1)[:, :2**16]
    trajectory = trajectory[:, :, :, :2**16] # shape = (Tsteps + 1, batch_size, 1, seq_len)
    
    wer.update(x0_recon, x0)
    srcs.update(x0_recon, x0)
    dnsmos.update(x0_recon)
    kad.update(x0_recon, x0)
        
    trajectory = trajectory.float().cpu()
    x0_recon = x0_recon.float().cpu()
    x0 = x0.float().cpu()
    x1 = x1.float().cpu()
    
    if i == 0:
        # just for debugging
        print("Mean x0:", x0.mean().item())
        print("Mean x1:", x1.mean().item())
        print("x0 shape:", x0.shape)
        print("x1 shape:", x1.shape)
        print("x0_recon shape:", x0_recon.shape)
        print("trajectory shape:", trajectory.shape)
    
    torch.save(
        {
        'x0': x0,
        'x1': x1,
        'x0_recon': x0_recon,
        'trajectory': trajectory
        },
        f"{folder_name}/batch_{i}.pt"
        )
    
params['timeschedule'] = timeschedule
params['sample_rate'] = sample_rate
torch.save(
    params,
    f"{folder_name}/params.pt"
)

metrics = {
    'wer': wer.values,
    'srcs': srcs.values,
    'dnsmos': dnsmos.values,
    'kad': kad.compute(),
}

torch.save(
    metrics,
    f"{folder_name}/metrics.pt"
)
