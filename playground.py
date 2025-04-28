from src.lightning_modules.dsb import PretrainedDSBModel
import torch
from src.dataset import EarsWHAMUnpaired
from src.callbacks.metrics import calculate_curvature_displacement
from src.callbacks.utils import get_batch_from_dataset

DUMMY = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PretrainedDSBModel('180425125453', 10)
model.to(device)
dataset = EarsWHAMUnpaired(length_seconds=2.23, sample_rate=16000, train=False, return_pair=True)
timeschedule = model.scheduler.gammas_bar

x0, x1 = get_batch_from_dataset(dataset, batch_size=32, shuffle=True)
x0, x1 = x0.to(device), x1.to(device)

# snr_levels = torch.linspace(5, 30, 32).long()
# x1 = [dataset.get_item(0, snr=snr.item())[1] for snr in snr_levels]
# x1 = torch.stack(x1, dim=0).to(device)

if DUMMY:
    x0_traj_encoded = torch.randn((101, x1.shape[0], 1)).to(device)
else:
    x1_encoded = model.encode(x1)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        x0_traj_encoded = model.sample(x1_encoded, forward=False, return_trajectory=True, show_progress=True, noise='none')

x0_traj_encoded = x0_traj_encoded.cpu()
C_t = calculate_curvature_displacement(x0_traj_encoded, timeschedule)
C_t_mean = C_t.mean(dim=0)
C_t_std = C_t.std(dim=0)

import matplotlib.pyplot as plt

yticks = [10 ** i for i in range(-2,1)]
plt.plot(timeschedule[1:], C_t_mean, label='Curvature Displacement')
plt.fill_between(timeschedule[1:], C_t_mean - C_t_std, C_t_mean + C_t_std, alpha=0.2, label='Std. Dev.')
plt.title('Curvature Displacement')
plt.yscale('log')
plt.xlabel('t')
plt.ylabel('Curvature Displacement')
plt.yticks(yticks, [str(i) for i in yticks])
plt.grid()
plt.legend()
plt.savefig('figures/curvature_displacement.png')