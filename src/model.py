from .utils import _bool
import utils

from argparse import ArgumentParser

import torch
from torch import nn
import numpy as np


def init_model_arg_parser(parser: ArgumentParser):
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Transform every audio to this sample rate",
    )
    parser.add_argument(
        "--generic_bands",
        type=_bool,
        default="false",
        help="Use generic band schema (refined BSRNN)",
    )
    parser.add_argument(
        "--chunk_size_in_seconds",
        type=int,
        default=1,
        help="split waveform into x second chunks",
    )
    parser.add_argument("--n_fft", type=int, default=2048, help="STFT's n_fft param")
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128 // 4,
        help="size of band-split module feature dim",
    )
    parser.add_argument(
        "--num_blstm_layers",
        type=int,
        default=24 // 4,
        help="number of stacked RNNs in BandSequence module",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=512 // 2,
        help="number of MaskEstimation module feature dim",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=95,
        help="NOTE: batch size is not songs but chunks in a single song",
    )

    parser.add_argument(
        "--part",
        type=str,
        default=[],
        dest="parts",
        help="which part to train on (not MTL)",
        action="append",
    )


class ModelConfig:
    sample_rate: int
    generic_bands: bool
    chunk_size_in_seconds: int
    n_fft: int
    feature_dim: int
    num_blstm_layers: int
    mlp_dim: int
    parts: list[str]

    def __init__(self, args: dict):
        self.sample_rate = args["--sample_rate"]
        self.generic_bands = args["--generic_bands"]
        self.chunk_size_in_seconds = args["--chunk_size_in_seconds"]
        self.n_fft = args["--n_fft"]
        self.feature_dim = args["--feature_dim"]
        self.num_blstm_layers = args["--num_blstm_layers"]
        self.mlp_dim = args["--mlp_dim"]
        self.parts = args["parts"]
        self.chunk_size = self.chunk_size_in_seconds * self.sample_rate
        self.win_length = self.n_fft
        self.hop_length = self.win_length // 4

    def create_model(self, num_sources: int) -> torch.nn.Module:
        return BSRNN(num_sources=num_sources, cfg=self)


class Vis:
    """Generate model's visual graph"""

    def visualize(self, input):
        pass
        # TODO:


def microtonal_notes(divisions_per_octave):
    A4_freq = 440.0
    min_freq = 20.0
    max_freq = 20000.0
    notes = []
    freqs = []
    current_freq = min_freq
    while current_freq <= max_freq:
        semitones_from_A4 = 12 * np.log2(current_freq / A4_freq)
        nearest_microtone = round(semitones_from_A4 * divisions_per_octave / 12)
        nearest_freq = A4_freq * (2 ** (nearest_microtone / divisions_per_octave))

        # Check if the frequency is a microtonal (not a standard semitone)
        if nearest_microtone % (divisions_per_octave // 12) != 0:
            note_name = utils.get_microtone_name(
                nearest_microtone / (divisions_per_octave / 12), divisions_per_octave
            )
            notes.append(note_name)
            freqs.append(nearest_freq)

        # Move to the next microtone
        current_freq = A4_freq * (2 ** ((nearest_microtone + 1) / divisions_per_octave))
    return (notes, freqs)


def create_evenly_distributed_splits(num_splits):
    _, freqs = microtonal_notes(24)
    splits = []
    last_freq = 0
    for freq in utils.evenly_skip_elements(freqs, num_splits):
        splits.append((freq, freq - last_freq))
        last_freq = freq
    return splits


# ~~~~~~~~~~~~~~~~~~~~~
# BandSplit
# ~~~~~~~~~~~~~~~~~~~~~

# Numbers are extracted from the paper
splits_generic = create_evenly_distributed_splits(41)
splits_v7 = [
    # below 1kh, bandwidth 100hz
    (1000, 100),
    # above 1kh and below 4khz, bandwidth 250hz
    (4000, 250),
    (8000, 500),
    (16000, 1000),
    (20000, 2000),
]


class BandSplit(nn.Module, Vis):
    def __init__(self, cfg: ModelConfig):
        super(BandSplit, self).__init__()

        self.splits = splits_generic if cfg.generic_bands else splits_v7

        #### Make splits

        # convert fft to freq
        freqs = cfg.sample_rate * torch.fft.fftfreq(cfg.n_fft)[: cfg.n_fft // 2 + 1]
        freqs[-1] = cfg.sample_rate // 2
        indices = []
        start_freq, start_index = 0, 0

        # create FC networks per bands
        # -- norm -> fc --> out
        # -- norm -> fc --> out
        # -- norm -> fc --> out
        for end_freq, step in self.splits:
            bands = torch.arange(start_freq + step, end_freq + step, step)
            start_freq = end_freq
            for band in bands:
                end_index = freqs[freqs < band].shape[0]
                if end_index != start_index or not cfg.generic_bands:
                    indices.append((start_index, end_index))
                start_index = end_index
        indices.append((start_index, freqs.shape[0]))
        self.band_indices = indices
        self.fully_connected_out = cfg.feature_dim

        import torchaudio.transforms as T

        self.temporal_dim = int(
            np.ceil(
                cfg.chunk_size
                / T.Spectrogram(
                    n_fft=cfg.n_fft,
                    win_length=cfg.win_length,
                    hop_length=cfg.hop_length,
                ).hop_length
            )
        )

        self.layer_norms = nn.ModuleList(
            [
                # * 2 is for added dim of view_as_real
                nn.LayerNorm([(band_end - band_start) * 2, self.temporal_dim])
                for band_start, band_end in self.band_indices
            ]
        )

        self.layer_fcs = nn.ModuleList(
            [
                # * 2 is for added dim of view_as_real
                nn.Linear((band_end - band_start) * 2, self.fully_connected_out)
                for band_start, band_end in self.band_indices
            ]
        )

    def forward(self, chunk_ftt):
        batch_size = chunk_ftt.size(0)
        stack = []
        for i, (band_start, band_end) in enumerate(self.band_indices):
            band = chunk_ftt[:, band_start:band_end, :]
            # band is shape of (B, F, T)
            band = torch.view_as_real(band)  # (B, F, T, 2)
            # convert to (B, 2, F, T) to be able to feed it to the norm
            band = band.permute(0, 3, 1, 2)

            # norm is (..., F, T) and fc is (Fxfully_connected_out)
            # we should make norm (..., T, F) in order to feed it to the fc
            band = band.reshape(batch_size, -1, band.size(-1))  # -1 = T
            norm = self.layer_norms[i](band)

            norm = norm.transpose(-1, -2).contiguous()
            fc_y = self.layer_fcs[i](norm)

            stack.append(fc_y)
        return torch.stack(stack, dim=1)


# ~~~~~~~~~~~~~~~~~~~~~
# BandSequence
# ~~~~~~~~~~~~~~~~~~~~~


class RNN(nn.Module):
    def __init__(self, input_dim_size):
        super(RNN, self).__init__()
        self.input_dim_size = input_dim_size
        # paper specified group norm
        self.norm = nn.ModuleList(
            [nn.GroupNorm(self.input_dim_size, self.input_dim_size) for _ in range(2)]
        )
        self.blstm = nn.ModuleList(
            [
                nn.LSTM(
                    self.input_dim_size,
                    self.input_dim_size,
                    bidirectional=True,
                    batch_first=True,
                )
                for _ in range(2)
            ]
        )
        self.fc = nn.ModuleList(
            [nn.Linear(self.input_dim_size * 2, self.input_dim_size) for _ in range(2)]
        )

    def forward(self, x):
        # input is b, bands(K), temporal_dim(t), input_dim_size

        # First loops converts the shape to [B, T, K, N]
        # and the second loop converts it back to [B, K, T, N]
        for i in range(2):
            B, K, T, N = x.shape
            out = x.view(B * K, T, N)
            out = self.norm[i](out.transpose(-1, -2)).transpose(-1, -2)
            out = self.blstm[i](out)[0]
            out = self.fc[i](out)
            x = out.view(B, K, T, N) + x
            x = x.permute(0, 2, 1, 3).contiguous()

        return x


class BandSequence(nn.Module, Vis):
    def __init__(self, band_split: BandSplit, cfg: ModelConfig):
        super(BandSequence, self).__init__()
        self.rnns = nn.Sequential(
            *[
                RNN(input_dim_size=band_split.fully_connected_out)
                for _ in range(cfg.num_blstm_layers)
            ]
        )

    def forward(self, x):
        # (bands, temporal_dim, fc_out)
        return self.rnns(x)


# ~~~~~~~~~~~~~~~~~~~~~
# MaskEstimation
# ~~~~~~~~~~~~~~~~~~~~~


class MaskEstimation(nn.Module, Vis):
    def __init__(self, band_split: BandSplit, cfg: ModelConfig):
        super(MaskEstimation, self).__init__()

        max_indice_diff = max([e - s for s, e in band_split.fully_connected_out])
        # TODO: support older version
        num_hiddens = lambda e, s: 3 * (max_indice_diff - (e - s) + 1)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(
                        [band_split.temporal_dim, band_split.fully_connected_out]
                    ),
                    nn.Linear(band_split.fully_connected_out, cfg.mlp_dim),
                    nn.Tanh(),
                    # double the output dim to use in GLU
                    # the extra *2 is for returning as complex
                    nn.Linear(cfg.mlp_dim, (e - s) * 2 * 2),
                    nn.GLU(),
                )
                for s, e in band_split.band_indices
            ]
        )

    def forward(self, x):
        # (b, k, temporal_dim, fc_out)
        parts = []
        for i in range(x.shape[1]):
            y = self.layers[i](x[:, i]).contiguous()
            B, T, F = y.shape
            y = y.permute(0, 2, 1).contiguous()  # B F T
            # basically halve the freq dim and use it for phasee
            y = y.view(B, 2, F // 2, T)  # (B, 2, F, T)
            y = y.permute(0, 2, 3, 1)  # (B, F, T, 2)
            y = torch.view_as_complex(y.contiguous())

            parts.append(y)

        # (b, f, t)
        return torch.cat(parts, dim=-2)


# ~~~~~~~~~~~~~~~~~~~~~
# BSRNN (combined module)
# ~~~~~~~~~~~~~~~~~~~~~


class BSRNN(nn.Module, Vis):
    def __init__(self, cfg: ModelConfig, num_sources=1):
        super(BSRNN, self).__init__()

        self.split = BandSplit(cfg)
        self.sequence = BandSequence(band_split=self.split, cfg=cfg)

        self.masks = nn.ModuleList(
            [MaskEstimation(band_split=self.split, cfg=cfg) for _ in range(num_sources)]
        )

    def forward(self, chunk_fft):
        # standard
        mean = chunk_fft.mean(dim=(1, 2), keepdim=True)
        std = chunk_fft.std(dim=(1, 2), keepdim=True)
        chunk_fft = (chunk_fft - mean) / (std + 1e-5)

        y = self.split(chunk_fft)
        y = self.sequence(y)
        masks = torch.stack([mask(y) for mask in self.masks], dim=0)

        # de-standard
        masks = (masks * std) + mean

        return masks
