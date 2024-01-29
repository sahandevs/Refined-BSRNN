from pprint import pprint
from utils import sep
from model import ModelConfig
import utils

import torch
import torchaudio

from pathlib import Path
from argparse import _SubParsersAction, ArgumentParser


def init_arg_parser(subparsers: _SubParsersAction):
    COMMAND_NAME = "infer"
    parser = subparsers.add_parser(COMMAND_NAME, help="infer mode")
    parser.add_argument(
        "--checkpoint_fp",
        type=str,
        required=True,
        help="location of checkpoint file (.pt)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="model input audio file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="directory to output source(s)",
    )

    return COMMAND_NAME, parser


class Config:
    checkpoint_fp: Path
    out_dir: Path
    input: Path

    def __init__(self, args: dict):
        self.checkpoint_fp = args["checkpoint_fp"]
        self.out_dir = args["out"]
        self.input = args["input"]


def do_infer(args: dict):
    cfg = Config(args)

    print("Infer mode Configuration:")
    pprint(cfg.__dict__)
    sep()

    cp = torch.load(cfg.checkpoint_fp)
    print("Checkpoint Configuration:")
    pprint(cp["cfg"].__dict__)
    sep()

    model = ModelConfig(cp["cfg"].__dict__).create_model(
        num_sources=len(cp["cfg"].__dict__["parts"])
    )
    model.load_state_dict(cp["model"])

    x = utils.load_audio(cfg.input, cfg=cp["cfg"])
    normal_waveform, gain_factor, peak_gain_factor = utils.normalize_waveform(
        x, cfg=cp["cfg"]
    )
    splits, padding_length = utils.split(normal_waveform, cfg=cp["cfg"])
    to_spectrogram = utils.to_spectrogram(cp["cfg"])
    from_spectrogram = utils.from_spectrogram(cp["cfg"])

    # TODO: use merge to create the full waveform
    split_stft = to_spectrogram(splits[0])
    y = model(split_stft.unsqueeze(0))

    prefix = cfg.input.name.split(".")[0]

    for i, part in enumerate(cp["cfg"].parts):
        # torch.Size([4, 1, 1025, 87])
        print(f"saving {part}")
        part_stft = y[i][0]
        wav = from_spectrogram(part_stft)
        torchaudio.save(
            cfg.out_dir / f"{prefix}-{part}.wav", wav, cp["cfg"].sample_rate
        )
