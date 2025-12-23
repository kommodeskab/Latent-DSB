from src.utils import config_from_id, get_ckpt_path

experiment_id = "301025072902"
config = config_from_id(experiment_id)

import hydra
model_config = config['model']
network = hydra.utils.instantiate(model_config['model'])
encoder_decoder = hydra.utils.instantiate(model_config['encoder_decoder'])
ckpt_path = get_ckpt_path(experiment_id, last=True)

from src.lightning_modules import DSB
dsb = DSB.load_from_checkpoint(ckpt_path, model=network, encoder_decoder=encoder_decoder)

dataset = hydra.utils.instantiate(config['data']['x1_valset'])

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = dsb.to(device)
sample = dataset[10].unsqueeze(0).to(device)
encoded = dsb.encoder_decoder.encode(sample)
with torch.no_grad():
    recon = dsb.sample(encoded, direction='backward', scheduler_type='cosine', return_trajectory=False, num_steps=10, verbose=True)
decoded = dsb.encoder_decoder.decode(recon)

decoded = decoded.flatten().cpu()
import torchaudio
torchaudio.save("reconstructed.wav", decoded.unsqueeze(0), sample_rate=16000)