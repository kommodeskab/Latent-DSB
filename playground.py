from src.networks.speech_separation import SpeechbrainSepformer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

model = SpeechbrainSepformer()

model.to(device)

