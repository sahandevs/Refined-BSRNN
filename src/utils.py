# boolean type for argparse
# source: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
_bool = lambda x: x.lower() == "true"


class GlobalConfig:
    seed: int
    device: str
    notebook: bool

    def __init__(self, dict):
        self.seed = dict["seed"]
        import random

        if self.seed == -1:
            self.seed = int(random.random() * 100)
        import torch
        import numpy

        torch.manual_seed(self.seed)
        numpy.random.seed(self.seed)

        self.device = dict["device"]
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)

        try:
            import IPython.display as idp

            self.notebook = True
        except:
            self.notebook = False


GLOBAL_CONFIG: GlobalConfig = None


def sep():
    """prints a separator"""
    import os

    print("―" * os.get_terminal_size().columns)


def count_model_parameters(model):
    """number of total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ~~~~Pre-process and Post-process utils~~~~

import torch
from torch import nn

import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt


def show_idp_audio(waveform, cfg):
    import IPython.display as idp

    n = 14
    return idp.display(
        idp.Audio(
            waveform,
            rate=cfg.sample_rate,
        )
    )


def load_audio(path, cfg):
    waveform, sr = torchaudio.load(path)
    # Convert everthing to mono channel for simplicity
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
        # waveform is now a vector
    # Resample everything to 44.1khz for simplicity
    resampler = T.Resample(sr, cfg.sample_rate, dtype=waveform.dtype)
    waveform = resampler(waveform)

    if GLOBAL_CONFIG.notebook:
        # samplerate = 1/t
        # display the first 3 seconds
        show_idp_audio(waveform, cfg)

    return waveform


def rms_normalize(waveform, target_rms):
    current_rms = torch.sqrt(torch.mean(waveform**2))
    gain_factor = target_rms / (current_rms + 1e-10)
    normalized_waveform = waveform * gain_factor
    return normalized_waveform, gain_factor


def rms_denormalize(normalized_waveform, gain_factor):
    inverse_gain = 1 / gain_factor
    reversed_waveform = normalized_waveform * inverse_gain
    return reversed_waveform


def peak_normalize(waveform, target_peak):
    peak_value = torch.max(torch.abs(waveform))
    peak_gain_factor = target_peak / (peak_value + 1e-10)
    normalized_waveform = waveform * peak_gain_factor
    return normalized_waveform, peak_gain_factor


def peak_denormalize(normalized_waveform, peak_gain_factor):
    inverse_peak_gain = 1 / peak_gain_factor
    reversed_waveform = normalized_waveform * inverse_peak_gain
    return reversed_waveform


def inspect_waveform(waveform, cfg):
    transform = T.Loudness(cfg.sample_rate)
    return f"LKFS:{transform(waveform.unsqueeze(0))} max: {waveform.max()} min: {waveform.min()} avg: {waveform.mean()}"


def normalize_waveform(waveform, cfg):
    """rms -> peak"""
    # target rms can be anything. the important part here
    # is to be constant for all kind of songs

    normalized_waveform, gain_factor = rms_normalize(waveform, target_rms=0.1)
    # setting target peak to 1.0 forces the values between -1.0 < y < 1.0
    normalized_waveform, peak_gain_factor = peak_normalize(
        normalized_waveform, target_peak=0.1
    )
    return normalized_waveform, gain_factor, peak_gain_factor


def de_normalize_waveform(waveform, gain_factor, peak_gain_factor, cfg):
    waveform = peak_denormalize(waveform, peak_gain_factor)
    waveform = rms_denormalize(waveform, gain_factor)
    return waveform


def split(
    waveform,
    cfg,
    drop_last=False,
):
    # we have a vector by length n and we want to split it to even chunks by length of
    # chunk_size
    padding_length = (
        cfg.chunk_size - waveform.shape[0] % cfg.chunk_size
    ) % cfg.chunk_size
    waveform = nn.functional.pad(waveform, (0, padding_length), "constant", 0)
    # -1 means automatically infer based on other dims
    chunked_waveform = waveform.view(-1, cfg.chunk_size)

    if GLOBAL_CONFIG.notebook:
        fig = plt.figure(constrained_layout=True, figsize=(16, 4))
        subfigs = fig.subfigures(2, 1).flat

        # first 3 chunk_size of waveform
        w = waveform[: 3 * cfg.chunk_size].detach().numpy()
        ylim = [w.max() * 1.1, w.min() * 1.1]

        def time_axis(start, duration):
            return (
                torch.arange(
                    start * cfg.sample_rate, (duration + start) * cfg.sample_rate
                )
                / cfg.sample_rate
            )

        axes = subfigs[0].subplots(1, 1)
        axes.plot(time_axis(0, 3), w, linewidth=0.3)
        axes.set_xlabel("time [s] for first 3 seconds")
        axes.set_ylim(ylim)

        # first 4 chunks + last chunk
        axes = subfigs[1].subplots(1, 5)
        for i, chunk in enumerate([0, 1, 3, 4, chunked_waveform.shape[0] - 1]):
            axes[i].plot(
                time_axis(0, cfg.chunk_size_in_seconds),
                chunked_waveform[chunk],
                linewidth=0.3,
            )
            axes[i].set_title(f"chunk {chunk}")
            axes[i].set_ylim(ylim)
    if drop_last:
        return chunked_waveform[:-1], 0
    return chunked_waveform, padding_length


def merge(chunks, padding_length):
    merged_waveform = torch.cat([torch.flatten(x) for x in chunks])
    return merged_waveform[:-padding_length]


def visualize_spectrogram(chunk, chunk_stft, cfg, title="Spectrogram"):
    import librosa

    _, axis = plt.subplots(2, 1, figsize=(16, 5))
    axis[0].imshow(
        librosa.power_to_db(chunk_stft.abs().detach().numpy() ** 2),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )
    axis[0].set_yscale("symlog")
    axis[0].set_title(title)
    if chunk is not None:
        axis[1].plot(chunk, linewidth=0.5)
        axis[1].grid(True)
        axis[1].set_xlim([0, len(chunk)])


def to_spectrogram(cfg):
    transform_spectrogram = T.Spectrogram(
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        window_fn=torch.hamming_window,
        power=None,
    )

    def inner(chunk, title=""):
        chunk_stft = transform_spectrogram(chunk)
        if GLOBAL_CONFIG.notebook:
            visualize_spectrogram(chunk, chunk_stft, title)

        return chunk_stft

    return inner


def from_spectrogram(cfg):
    transform_inv_spectrogram = T.InverseSpectrogram(
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        window_fn=torch.hamming_window,
    )

    def inner(chunk_stft):
        chunk = transform_inv_spectrogram(chunk_stft)

        if GLOBAL_CONFIG.notebook:
            visualize_spectrogram(chunk.detach().numpy(), chunk_stft, cfg=cfg)

        return chunk

    return inner


def get_microtone_name(semitones_from_A4, divisions_per_octave):
    """get musical note name from freq"""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    microtone_index = (
        int((semitones_from_A4 + 9) * divisions_per_octave / 12) % divisions_per_octave
    )
    octave = int((semitones_from_A4 + 9) // 12)
    note_index = microtone_index // (divisions_per_octave // 12)
    microtone_suffix = f"+{microtone_index % (divisions_per_octave // 12)}"
    return (
        notes[note_index]
        + (microtone_suffix if microtone_suffix != "+0" else "")
        + str(octave)
    )


def evenly_skip_elements(input_list, k):
    step = len(input_list) / k
    new_list = [input_list[int(i * step)] for i in range(k)]

    return new_list
