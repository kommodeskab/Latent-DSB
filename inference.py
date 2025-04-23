#%%
from src.networks import UNet50
from src.networks.encoders import HifiGan
from src.lightning_modules import DSB
from src.utils import get_ckpt_path
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
forward = UNet50(in_channels=1, out_channels=1)
backward = UNet50(in_channels=1, out_channels=1)
encoder = HifiGan()
ckpt_path = get_ckpt_path('180425125453', last=False, filename='DSB_iteration_10.ckpt')
dsb = DSB.load_from_checkpoint(ckpt_path, forward_model=forward, backward_model=backward, encoder_decoder=encoder)
dsb.to(device)
#%%
from src.dataset import EarsWHAMUnpaired
from src.callbacks.utils import get_batch_from_dataset
dataset = EarsWHAMUnpaired(length_seconds=4.47, sample_rate=16000, train=False, return_pair=True)
x0, x1 = get_batch_from_dataset(dataset, 16, shuffle=True)
x0 = x0.to(device)
x1 = x1.to(device)
#%%
import torchaudio
import os
os.makedirs('test_results', exist_ok=True)

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    x1_encoded = dsb.encode(x1)
    print(f"Encoded shape: {x1_encoded.shape}")
    x0_recon_encoded = dsb.sample(x1_encoded, forward=False, show_progress=True, noise='inference')
    x0_recon = dsb.decode(x0_recon_encoded)

for i in range(x0.shape[0]):
    # concat x0, x1 and x0_recon along the time dimension
    x = torch.cat([x0[i].cpu().flatten(), x1[i].cpu().flatten(), x0_recon[i].cpu().flatten()], dim=0)
    # save to file
    torchaudio.save(f'test_results/{i}.wav', x.unsqueeze(0), 16000)
    
# %%
