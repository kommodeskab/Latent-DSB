from src.networks.simple_models import SimpleNetwork, SimpleNetworkImages
from src.networks.encoders import IdentityEncoderDecoder, BaseEncoderDecoder, STFTEncoderDecoder, HifiGan, StableDiffusionXL, PolarSTFTEncoderDecoder, OpenSoundEncoder, PretrainedVAENetwork
from src.networks.gfb import STFTbackbone, AltSTFTbackbone
from src.networks.dhariwal import UNetModel
from src.networks.encoders import VAENetwork
from src.networks.unet_1d import UNet1D
from src.networks.huggingface import HuggingfaceUNet1D
from src.networks.transformer import ConditionalTimeSeriesTransformer