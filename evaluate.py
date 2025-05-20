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
from GFB import GFB
import torchaudio

torch.manual_seed(0)
# make an argument parser for experiment_name
parser = ArgumentParser()
parser.add_argument('--experiment_id', type=str, default='180425125453')
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--length', type=float, default=2.23)
parser.add_argument('--sdr', type=float, default=None)
parser.add_argument('--baseline', default=False)
args = parser.parse_args()

experiment_id = args.experiment_id
num_samples = args.num_samples
batch_size = args.batch_size
length_seconds = args.length
sdr = args.sdr
baseline = args.baseline # test baseline, i.e. identity mapping

for arg, value in vars(args).items():
    print(f"{arg}: {value}")

print("Evaluating ...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") 

if experiment_id == 'GFB':
    # for validation of Gaussian Flow Bridge (GFB)
    dsb = GFB(device)
    seq_length = 65536
    length_seconds = seq_length / 16000
    x1_dataset = ClippedVCTK(length_seconds, sample_rate=16000, train=False)
    print(f"Using GFB model with seq_length: {seq_length} and length_seconds: {length_seconds}")
else:
    dsb_iterations = get_dsb_iterations(experiment_id)
    dsb_iteration = max(dsb_iterations)

    dsb = load_dsb_model(experiment_id, dsb_iteration)
    dsb = dsb.to(device)    

    _, x1_dataset = load_dsb_datasets(experiment_id)
    
    x1_dataset : EarsWHAMUnpaired | ClippedLibri | ClippedVCTK
    x1_dataset.set_length(length_seconds)
    
if baseline:
    experiment_id = 'baseline'
    
x1_dataset.return_pair = True
sample_rate = x1_dataset.sample_rate

if sdr is not None:
    if isinstance(x1_dataset, (ClippedVCTK, ClippedLibri)):
        db = x1_dataset.what_db_for_sdr(sdr)
        x1_dataset.noise_range = (db, db)
        print(f"Setting noise range to {x1_dataset.noise_range} dB for SDR {sdr} dB")
    elif isinstance(x1_dataset, EarsWHAMUnpaired):
        x1_dataset.noise_range = (sdr, sdr)
        print(f"Setting noise range to {x1_dataset.noise_range} SDR for SDR {sdr} dB")

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
    
    if baseline:
        x0_recon = x1
    else:
        dtype = torch.float16 if experiment_id != 'GFB' else torch.float32
        with torch.autocast(device_type=device, dtype=dtype):
            x1_encoded = dsb.encode(x1)
            x0_recon_encoded = dsb.sample(x1_encoded, forward=False, show_progress=True, noise='inference')
            x0_recon = dsb.decode(x0_recon_encoded).float() # convert from bfloat16 to float32
    
    if i == 0:
        # sanity check the results by saving the first audio sample
        # concat x0_recon, x0 and x1 along the time dimension to make a single audio file
        sanity_check = torch.cat([x1[0], x0_recon[0], x0[0]]).cpu().view(1, -1)
        torchaudio.save(f"test_results/sanity_check_{experiment_id}.wav", sanity_check, sample_rate=sample_rate)
        
    for name, metric in metrics.items():
        metric : DeepNoiseSuppressionMeanOpinionScore | ScaleInvariantSignalDistortionRatio | PerceptualEvaluationSpeechQuality | WER
        
        if name in ['mos']:
            metric.update(x0_recon.squeeze(1))
            
        if name in ['sisdr', 'wer']:
            metric.update(x0_recon.squeeze(1), x0.squeeze(1))
            
        if name in ['pesq']:
            try:
                metric.update(x0_recon.squeeze(1), x1.squeeze(1))
            except Exception as e:
                # pesq throws an error if no speech is detected
                print(f"Error in PESQ calculation (batch {i}): {e}")
                continue
            
        if name == 'srcs':
            metric.update(x0_recon.squeeze(1), x0.squeeze(1))
    
        
for name, metric in metrics.items():
    print(f"{name}: {metric.compute()}")