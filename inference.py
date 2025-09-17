import torch
from argparse import ArgumentParser
from src.lightning_modules.dsb import get_dsb_iterations, load_dsb_model
from GFB import GFB
from SPADE import SPADE
from WPE import WPE
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from src.callbacks.metrics import WER, SRCS, DNSMOS, KAD, SISDRi, MelCepstralDistance
import random
import numpy as np
from torch import Tensor
from src.lightning_modules.esdsb import load_esdsb_model
from src.lightning_modules import DSB, ESDSB
from src.networks.speech_separation import SpeechbrainSepformer, AsteroidConvTasNet

schedule = 'cosine'
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
parser.add_argument('--num_steps', type=int, required=True)
parser.add_argument('--noise_factor', type=float, default=1.0)
parser.add_argument('--dsb_iteration', type=int, default=None)
parser.add_argument('--what_test', type=str, required=True) # clip, rir, noise
parser.add_argument('--folder_name', type=str, required=True)
parser.add_argument('--save_trajectory', default=False, help="Whether to save the trajectory or not")

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
save_trajectory : bool = args.save_trajectory

assert what_test in ['rir', 'clip', 'noise'], "what_test must be either 'rir' or 'clip'"

folder_name = f"test_results/{folder_name}"
os.makedirs(folder_name, exist_ok=True)

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

if experiment_id == 'GFB':
    print("Using GFB model")
    assert dsb_iteration is None, "DSB iteration is not supported for GFB"
    assert num_steps > 1, "GFB cant do one-step inference"
    
    length_seconds = 4.096
    
    dsb = GFB(device, what_test=what_test, schedule_type=schedule)
    # Tsteps is the number of steps for encoding / decoding, total num_steps = 2 * Tsteps
    dsb.Tsteps = num_steps // 2
    
    timeschedule = dsb.diff.get_schedule(dsb.Tsteps, end_t=1.0, type=schedule)
    
elif experiment_id == 'sepformer':
    print("Using SpeechbrainSepformer model")
    dsb = SpeechbrainSepformer()
    length_seconds = 4.096
    # sample_rate = dsb.sample_rate
    timeschedule = None
    
elif experiment_id == 'convtasnet':
    print("Using AsteroidConvTasNet model")
    dsb = AsteroidConvTasNet()
    length_seconds = 4.096
    # sample_rate = dsb.sample_rate
    timeschedule = None

elif experiment_id == 'baseline':
    print("Using baseline model")
    assert dsb_iteration is None, "DSB iteration is not supported for baseline"

    length_seconds = 4.096
    dsb = BaselineModel(num_steps=num_steps)
    timeschedule = None
    
elif 'ESDSB' in experiment_id:
    print(f'Using ESDSB model with experiment_id: {experiment_id}')
    _, experiment_id = experiment_id.split('_')
    
    dsb = load_esdsb_model(experiment_id)
    dsb.scheduler.noise_factor = noise_factor
    
    from src.networks import STFTEncoderDecoder, HifiGan
    if isinstance(dsb.encoder_decoder, STFTEncoderDecoder):
        length_seconds = 4.096
    elif isinstance(dsb.encoder_decoder, HifiGan):
        length_seconds = 4.47
    else:
        raise ValueError(f"Unknown encoder_decoder type: {type(dsb.encoder_decoder)}")

    timeschedule = dsb.scheduler.get_timeschedule(num_steps=num_steps, scheduler_type=schedule, direction='backward')

elif experiment_id == 'WPE':
    print("Using WPE model")
    dsb = WPE()
    length_seconds = 4.096
    timeschedule = None

elif experiment_id == 'SPADE':
    print("Using SPADE model")
    dsb = SPADE()
    length_seconds = 4.096
    timeschedule = None

else:
    print(f"Using DSB model with experiment_id: {experiment_id}")
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

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, what_test : str):
        super().__init__()
        assert what_test in ['clip', 'rir', 'noise']
        self.path = f"/work3/s214630/data/{what_test}"
        
    def __len__(self) -> int:
        return num_samples
    
    def __getitem__(self, idx : int) -> tuple[Tensor, Tensor]:
        path = self.path + f"/batch_{idx}"
        data = torch.load(path, weights_only=True)
        return data['x0'], data['x1']
    
dataset = TestDataset(what_test)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

wer = WER(sample_rate, device)
srcs = SRCS(sample_rate, device)
dnsmos = DNSMOS(sample_rate, device)
kad = KAD(sample_rate, device)
sisdri = SISDRi()
mcd = MelCepstralDistance(sample_rate)

for i, batch in enumerate(tqdm(dataloader, desc="Loading batches")):
    x0, x1 = batch
    x0 : Tensor
    x1 : Tensor
    
    seq_len = int(length_seconds * sample_rate)
    x0, x1 = x0.to(device), x1.to(device)
    x0 = x0[:, :, :seq_len]
    x1 = x1[:, :, :seq_len]
    
    x1_encoded = dsb.encode(x1)
    
    if isinstance(dsb, ESDSB):
        trajectory = dsb.sample(x1_encoded, direction='backward', scheduler_type=schedule, num_steps=num_steps, return_trajectory=True)
    elif isinstance(dsb, (DSB, GFB, BaselineModel)):
        trajectory = dsb.sample(x1_encoded, forward=False, return_trajectory=True, noise='inference', noise_factor=noise_factor, show_progress=True)
    elif isinstance(dsb, (SpeechbrainSepformer, AsteroidConvTasNet)):
        trajectory = dsb.separate(x1).unsqueeze(0) # pesudo trajectory with a single time step
    elif isinstance(dsb, SPADE):
        trajectory = dsb.declip(x1).unsqueeze(0)
    elif isinstance(dsb, WPE):
        trajectory = dsb.dereverb(x1).unsqueeze(0)

    x0_recon_encoded = trajectory[-1]
    x0_recon = dsb.decode(x0_recon_encoded)
    
    # squeeze and make sure that length is no longer than 2**16 for fair comparison
    x0_recon = x0_recon.squeeze(1)[:, :2**16]
    x0 = x0.squeeze(1)[:, :2**16]
    x1 = x1.squeeze(1)[:, :2**16]
    trajectory = trajectory[:, :, :, :2**16]
    
    wer.update(x0_recon, x0)
    srcs.update(x0_recon, x0)  
    dnsmos.update(x0_recon)
    kad.update(x0_recon, x0)
    sisdri.update(x0, x1, x0_recon)
    mcd.update(x0_recon, x0)

    trajectory = trajectory.float().cpu()
    x0_recon = x0_recon.float().cpu()
    x0 = x0.float().cpu()
    x1 = x1.float().cpu()
    
    if not save_trajectory:
        # to save space
        print("Skipping saving trajectory as per user request...")
        trajectory = None
    
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
    'sisdri': sisdri.values,
    'mcd': mcd.values
}

print("Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")

torch.save(
    metrics,
    f"{folder_name}/metrics.pt"
)
