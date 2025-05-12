import torch
from torchmetrics.audio.dnsmos import DeepNoiseSuppressionMeanOpinionScore
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.callbacks.metrics import WER, SRCS
from argparse import ArgumentParser
from src.lightning_modules.dsb import get_dsb_iterations, load_dsb_datasets, load_dsb_model
from torch.utils.data import Subset
from src.dataset import EarsWHAMUnpaired, ClippedVCTK, ClippedLibri

torch.manual_seed(0)
# make an argument parser for experiment_name
parser = ArgumentParser()
parser.add_argument('--experiment_id', type=str, default='180425125453')
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dummy', type=bool, default=False)
parser.add_argument('--length', type=float, default=2.23)
parser.add_argument('--snr', type=float, default=None)
args = parser.parse_args()

experiment_id = args.experiment_id
num_samples = args.num_samples
batch_size = args.batch_size
dummy = args.dummy
length_seconds = args.length
snr = args.snr

for arg, value in vars(args).items():
    print(f"{arg}: {value}")

print("Evaluating ...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 

dsb_iterations = get_dsb_iterations(experiment_id)
dsb_iteration = max(dsb_iterations)

if not dummy:
    dsb = load_dsb_model(experiment_id, dsb_iteration)
    dsb = dsb.to(device)    

x0_dataset, x1_dataset = load_dsb_datasets(experiment_id)
x1_dataset : EarsWHAMUnpaired | ClippedLibri | ClippedVCTK
x1_dataset.set_length(length_seconds)
x1_dataset.return_pair = True
sample_rate = x1_dataset.sample_rate

if snr is not None:
    if isinstance(x1_dataset, (ClippedVCTK, ClippedLibri)):
        db = x1_dataset.what_db_for_snr(snr)
        x1_dataset.noise_range = (db, db)
        print(f"Setting noise range to {x1_dataset.noise_range} dB for SNR {snr} dB")
    elif isinstance(x1_dataset, EarsWHAMUnpaired):
        x1_dataset.noise_range = (snr, snr)
        print(f"Setting noise range to {x1_dataset.noise_range} snr for SNR {snr} dB")

# change the size of x1_dataset to num_samples, x1_dataset is of type torch.utils.data.Dataset
x1_dataset = Subset(x1_dataset, range(num_samples))
dataloader = DataLoader(x1_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

metrics = {
    'mos': DeepNoiseSuppressionMeanOpinionScore(fs=sample_rate, personalized=False, device=device),
    # 'sisdr': ScaleInvariantSignalDistortionRatio().to(device),
    # 'pesq': PerceptualEvaluationSpeechQuality(fs=sample_rate, mode='nb'),
    'wer': WER(sample_rate=sample_rate, device=device),
    'srcs': SRCS(sample_rate=sample_rate, device=device)
}

for i, batch in enumerate(tqdm(dataloader, desc="Evaluating..", unit="batch")):
    x0, x1 = batch
    x0 : torch.Tensor
    x1 : torch.Tensor
    
    x0, x1 = x0.to(device), x1.to(device)
    
    if dummy:
        x0_recon = torch.randn_like(x0) # TODO: replace with model output
    else:
        with torch.autocast(device_type=device, dtype=torch.float16):
            x1_encoded = dsb.encode(x1)
            x0_recon_encoded = dsb.sample(x1_encoded, forward=False, show_progress=True, noise='inference')
            x0_recon = dsb.decode(x0_recon_encoded).float() # convert from bfloat16 to float32
        
        if i == 0:
            # sanity check the results by saving the first audio sample
            import torchaudio
            torchaudio.save("test_results/sanity_check_sample.wav", x0_recon[0].cpu(), sample_rate=sample_rate)
            torchaudio.save("test_results/sanity_check_target.wav", x0[0].cpu(), sample_rate=sample_rate)
            torchaudio.save("test_results/sanity_check_input.wav", x1[0].cpu(), sample_rate=sample_rate)
            
    for name, metric in metrics.items():
        metric : DeepNoiseSuppressionMeanOpinionScore | ScaleInvariantSignalDistortionRatio | PerceptualEvaluationSpeechQuality | WER
        
        if name == 'mos':
            metric.update(x0_recon.squeeze(1))
            
        if name in ['sisdr', 'wer']:
            metric.update(x0_recon.squeeze(1), x0.squeeze(1))
            
        if name == 'pesq':
            try:
                metric.update(x0_recon.squeeze(1), x1.squeeze(1))
            except Exception as e:
                print(f"Error in PESQ calculation (batch {i}): {e}")
                continue
            
        if name == 'srcs':
            metric.update(x0_recon.squeeze(1), x0.squeeze(1))
    
        
for name, metric in metrics.items():
    print(f"{name}: {metric.compute()}")