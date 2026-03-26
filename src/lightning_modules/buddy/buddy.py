from src.lightning_modules.buddy.testing.operators.subband_filtering import BlindSubbandFiltering
from src.lightning_modules.buddy.testing.EulerHeunSamplerDPS import EulerHeunSamplerDPS
from src.lightning_modules.buddy.networks.ncsnpp import NCSNppTime
import src.lightning_modules.buddy.utils.training_utils as tr_utils

import os
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from src.lightning_modules import BaseLightningModule
from torch import Tensor
from dotenv import load_dotenv
import gdown

load_dotenv()


class Buddy(BaseLightningModule):
    """
    BUDDy: Single-channel Blind Unsupervised Dereverberation with Diffusion Models
    Taken from here: https://github.com/sp-uhh/buddy
    Used as a baseline
    """

    def __init__(self, num_steps: int = 201):
        super().__init__()

        self.args = OmegaConf.create(
            {
                "dset": {
                    "test": {
                        "_target_": "datasets.vctk.RealTestSamples",
                        "segment_length": -1,
                        "fs": 16000,
                        "path": "audio_examples",
                        "speakers_discard": [],
                        "speakers_test": ["p226", "p287"],
                        "normalize": False,
                        "num_examples": 2,
                        "shuffle": False,
                    }
                },
                "network": {
                    "_target_": "networks.ncsnpp.NCSNppTime",
                    "stft": {"n_fft": 510, "hop_length": 128, "center": True},
                    "nonlinearity": "swish",
                    "nf": 128,
                    "ch_mult": [1, 2, 2, 2],
                    "num_res_blocks": 1,
                    "attn_resolutions": [0],
                    "resamp_with_conv": True,
                    "time_conditional": True,
                    "fir": False,
                    "fir_kernel": [1, 3, 3, 1],
                    "skip_rescale": True,
                    "resblock_type": "biggan",
                    "progressive": "output_skip",
                    "progressive_input": "input_skip",
                    "progressive_combine": "sum",
                    "init_scale": 0,
                    "fourier_scale": 16,
                    "image_size": 256,
                    "embedding_type": "fourier",
                    "input_channels": 2,
                    "spatial_channels": 1,
                    "dropout": 0,
                    "centered": True,
                    "discriminative": False,
                },
                "diff_params": {
                    "_target_": "src.lightning_modules.buddy.diff_params.edm.EDM",
                    "type": "ve_karras",
                    "sde_hp": {"sigma_data": 0.05, "sigma_min": 1e-05, "sigma_max": 10, "rho": 10},
                },
                "tester": {
                    "name": "blind_dereverberation_buddy",
                    "tester": {"_target_": "testing.tester.Tester"},
                    "sampler": {"_target_": "testing.EulerHeunSamplerDPS.EulerHeunSamplerDPS"},
                    "modes": ["blind_dereverberation"],
                    "checkpoint": "VCTK_16k_4s_time-190000.pt",
                    "sampling_params": {
                        "same_as_training": False,
                        "sde_hp": {"sigma_data": 0.05, "sigma_min": 0.0001, "sigma_max": 0.5, "rho": 10},
                        "Schurn": 50,
                        "Snoise": 1,
                        "Stmin": 0,
                        "Stmax": 10,
                        "order": 1,
                        "T": num_steps,
                        "schedule": "edm",
                    },
                    "posterior_sampling": {
                        "zeta": 0.5,
                        "rec_loss": {
                            "name": "l2_comp_stft_summean",
                            "weight": 512,
                            "frequency_weighting": "none",
                            "compression_factor": 0.667,
                            "multiple_compression_factors": False,
                        },
                        "rec_loss_params": {
                            "name": "l2_comp_stft_summean",
                            "weight": 512,
                            "frequency_weighting": "none",
                            "compression_factor": 0.667,
                            "multiple_compression_factors": False,
                            "compression_factors": [1, 0.1],
                            "weights": [1, 0.1],
                        },
                        "RIR_noise_regularization": {
                            "use": True,
                            "crop_sigma_max": 0.01,
                            "crop_sigma_min": 0.0005,
                            "loss": {
                                "name": "l2_comp_stft_summean",
                                "weight": 2560,
                                "frequency_weighting": "none",
                                "compression_factor": 0.667,
                                "multiple_compression_factors": False,
                            },
                        },
                        "project_parameters": True,
                        "normalization_type": "grad_norm",
                        "blind_hp": {
                            "optimizer": "adam",
                            "lr_op": 0.1,
                            "beta1": 0.9,
                            "beta2": 0.99,
                            "noise": 0.1,
                            "lr_op_phase": 1,
                            "weight_decay": 0,
                            "op_updates_per_step": 10,
                            "grad_clip": 1,
                        },
                        "warm_initialization": {
                            "mode": "wpe_scaled",
                            "scaling_factor": 0.05,
                            "wpe": {"delay": 2, "taps": 50, "iterations": 5},
                        },
                        "constraint_speech_magnitude": {"use": True, "speech_scaling": 0.05},
                    },
                    "unconditional": {"num_samples": 1, "audio_len": 65536},
                    "informed_dereverberation": {
                        "path_RIRs": "...",
                        "files": "...",
                        "operator": "subband_filtering",
                        "name_params": ["T60s", "weights"],
                        "op_hp": {
                            "fix_EQ_extremes": True,
                            "NFFT": 1024,
                            "win_length": 512,
                            "hop": 128,
                            "window": "hann",
                            "Nf": 100,
                            "EQ_freqs": [
                                0,
                                125,
                                250,
                                375,
                                500,
                                625,
                                750,
                                875,
                                1000,
                                1250,
                                1500,
                                1750,
                                2000,
                                2250,
                                2500,
                                2750,
                                3000,
                                3500,
                                4000,
                                4500,
                                5000,
                                5500,
                                6000,
                                6500,
                                7000,
                                7500,
                                8000,
                            ],
                            "init_single_value": True,
                            "init_params": {"T60_breakpoints": [0.1], "multiexp_weighting": [2]},
                            "init_phases": "random_coherent",
                            "minimum_phase": True,
                            "fix_direct_path": True,
                            "num_GL_iter": 1,
                            "cumulative_decays": False,
                            "decay_scale": 1,
                            "Amin": 0,
                            "Amax": 40,
                            "T60min": 0.1,
                            "T60max": 2,
                            "clamp_A": True,
                            "clamp_decay": True,
                            "strictly_decreasing_decay": False,
                            "enforce_long_decay_in_second_exponential": True,
                            "n_iter_PR": 5,
                        },
                    },
                    "blind_dereverberation": {
                        "operator": "subband_filtering",
                        "test_params": {
                            "T60_breakpoints": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],
                            "multiexp_weighting": [[2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        },
                    },
                },
                "exp": {
                    "exp_name": "VCTK_16k_4s_time",
                    "model_dir": "/home/andbagge/buddy/experiments/buddy_wpe-init_noise-prior_N-201_rir-aligned_1exp",
                    "trainer": {"_target_": "training.trainer.Trainer"},
                    "optimizer": {"_target_": "torch.optim.Adam", "lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-08},
                    "lr_rampup_it": 10000,
                    "scheduler_step_size": 60000,
                    "scheduler_gamma": 0.8,
                    "batch_size": 16,
                    "num_workers": 4,
                    "seed": 1,
                    "resume": True,
                    "resume_checkpoint": "None",
                    "sample_rate": 16000,
                    "audio_len": 65536,
                    "ema_rate": 0.9999,
                    "ema_rampup": 10000,
                    "use_grad_clip": True,
                    "max_grad_norm": 1,
                    "restore": False,
                    "checkpoint_id": "None",
                },
                "logging": {
                    "log": True,
                    "log_interval": 1000,
                    "heavy_log_interval": 10000,
                    "save_model": True,
                    "save_interval": 10000,
                    "remove_old_checkpoints": True,
                    "num_sigma_bins": 20,
                    "print_model_summary": False,
                    "profiling": {"enabled": True, "wait": 5, "warmup": 10, "active": 2, "repeat": 1},
                    "log_spectrograms": False,
                    "stft": {"win_size": 1024, "hop_size": 256},
                    "wandb": {"entity": "eloimoliner", "project": "audiodps"},
                },
                "model_dir": "/home/andbagge/buddy/experiments/buddy_wpe-init_noise-prior_N-201_rir-aligned_1exp",
                "gpu": 0,
            }
        )
        data_path = os.getenv("DATA_PATH")
        url = "https://drive.google.com/uc?id=1j-BHDqiKxbEfpDUpc1GHZXeoHIA1nIal"
        output = f"{data_path}/model.pt"

        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        self.network = NCSNppTime(**self.args.network)
        self.diff_params = instantiate(self.args.diff_params)

        ckpt = torch.load(output, map_location="cpu", weights_only=False)
        tr_utils.load_state_dict(state_dict=ckpt, ema=self.network)

        self.sampler = EulerHeunSamplerDPS(
            model=self.network,
            diff_params=self.diff_params,
            args=self.args,
        )

    def common_step(self, batch, batch_idx):
        return ...

    def sample(self, x_start: Tensor, **kwargs) -> Tensor:
        output = []

        for x in x_start:
            x: Tensor = self.args.tester.posterior_sampling.warm_initialization.scaling_factor * x / x.std()
            with torch.no_grad():
                operator = BlindSubbandFiltering(
                    op_hp=self.args.tester.informed_dereverberation.op_hp, sample_rate=16000
                )
                operator.update_H(use_noise=True)

            with torch.enable_grad():
                pred = self.sampler.predict_conditional(x, operator=operator, shape=(1, x.shape[-1]), blind=True)
            output.append(pred)

        return torch.stack(output, dim=0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buddy = Buddy(num_steps=10).to(device)
    buddy.eval()
    for param in buddy.parameters():
        param.requires_grad = False
    with torch.no_grad():
        x = torch.randn(2, 1, 65536).to(device)
        out = buddy.sample(x)
    print(out.shape)
