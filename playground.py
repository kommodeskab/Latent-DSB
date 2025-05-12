import torch
from src.networks import UNet1D50

model = UNet1D50(in_channels=1, out_channels=1)
device = torch.device('cuda')

model.to(device)
x = torch.randn(128, 1, 224).to(device)
t = torch.randint(0, 100, (128,)).to(device)

out = model(x, t)

from tqdm import tqdm

for _ in tqdm(range(500)):
    with torch.autocast(device_type=device.type, dtype=torch.float16, cache_enabled=False):
        out = model(x, t)
    