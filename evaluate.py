from src.dataset import EarsWHAMUnpaired
import torch
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.callbacks.metrics import WER
from typing import Literal
from src.networks import UNet50
from src.networks.encoders import HifiGan
from src.lightning_modules import DSB
from src.utils import get_ckpt_path

experiment_id = '180425125453'
dsb_iteration = 10
test = False
batch_size = 16
max_batches = None

print("Evaluating ...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 
sample_rate = 16000

if not test:
    forward = UNet50(in_channels=1, out_channels=1)
    backward = UNet50(in_channels=1, out_channels=1)
    encoder = HifiGan()
    ckpt_path = get_ckpt_path(experiment_id, last=False, filename=f'DSB_iteration_{dsb_iteration}.ckpt')
    dsb = DSB.load_from_checkpoint(ckpt_path, forward_model=forward, backward_model=backward, encoder_decoder=encoder)
    dsb.to(device)

dataset = EarsWHAMUnpaired(length_seconds=4.47, sample_rate=sample_rate, train=False, return_pair=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

metrics = {
    'mos': DeepNoiseSuppressionMeanOpinionScore(fs=sample_rate, personalized=False, device=device),
    'sisdr': ScaleInvariantSignalDistortionRatio().to(device),
    'pesq': PerceptualEvaluationSpeechQuality(fs=sample_rate, mode='nb'),
    'wer': WER(sample_rate=sample_rate, device=device)
}

for i, batch in enumerate(tqdm(dataloader, desc="Evaluating..", unit="batch")):
    if max_batches is not None:
        if i >= max_batches:
            break
    
    speech, noisy_speech = batch
    speech : torch.Tensor
    noisy_speech : torch.Tensor
    
    speech, noisy_speech = speech.to(device), noisy_speech.to(device)
    
    if test:
        speech_recon = torch.randn_like(speech) # TODO: replace with model output
    else:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            x1_encoded = dsb.encode(noisy_speech)
            speech_recon_encoded = dsb.sample(x1_encoded, forward=False, show_progress=True, noise='inference')
            speech_recon = dsb.decode(speech_recon_encoded).float()
            
    for name, metric in metrics.items():
        if name == 'mos':
            metric.update(speech_recon.squeeze(1))
            
        if name in ['sisdr', 'wer']:
            metric.update(speech_recon.squeeze(1), speech.squeeze(1))
            print(metric.real_transcriptions)
            print(metric.generated_transcriptions)
            
        if name == 'pesq':
            try:
                metric.update(speech_recon.squeeze(1), noisy_speech.squeeze(1))
            except Exception as e:
                print(f"Error in PESQ calculation (batch {i}): {e}")
                continue
    
        
for name, metric in metrics.items():
    print(f"{name}: {metric.compute()}")