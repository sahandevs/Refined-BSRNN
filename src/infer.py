from pprint import pprint
from .utils import GLOBAL_CONFIG, sep
from .model import BSRNN, ModelConfig
import utils

import torch

from pathlib import Path
from argparse import _SubParsersAction, ArgumentParser


def init_arg_parser(subparsers: _SubParsersAction[ArgumentParser]):
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
        self.checkpoint_fp = args["--checkpoint_fp"]
        self.out_dir = args["--out_dir"]
        self.input = args["--input"]


def do_infer(parser: ArgumentParser):
    cfg = Config(parser.parse_args())

    print("Infer mode Configuration:")
    pprint(cfg.__dict__)
    sep()

    cp = torch.load(cfg.checkpoint_fp)
    print("Checkpoint Configuration:")
    pprint(cp.__dict__)
    sep()

    model = Inference(bsrnn=cp["model"], cfg=cp["cfg"])
    utils.load_audio(cfg.input, cfg=cp["cfg"])

    # TODO: save


class Inference(torch.nn.Module):
    def __init__(self, bsrnn: BSRNN, cfg: ModelConfig):
        super(Inference, self).__init__()
        self.cfg = cfg
        self.to_spectrogram = utils.to_spectrogram(cfg)
        self.from_spectrogram = utils.from_spectrogram(cfg)
        self.bsrnn = BSRNN()

    def forward(self, waveform):
        """Waveform in -> Waveform out :)"""

        # 1) normalize
        # 2) split
        # 3) feed to bsrnn
        # 4) convert spectogram to audio
        # 5) merge all splits
        # 6) de-normalize

        # TODO: implement with the new multi part framework
        normal_waveform, gain_factor, peak_gain_factor = utils.normalize_waveform(
            waveform
        )
        splits, padding_length = utils.split(normal_waveform, cfg=self.cfg)
        masked_splits = [[] for _ in range(len(splits))]
        for i, x_split in enumerate(splits):
            split_stft = self.to_spectrogram(x_split)
            masks = self.bsrnn(split_stft.unsqueeze(0))[0]

            for source in masks:
                wave = self.from_spectrogram(masked_complex)
                masked_splits[i].append(wave)

        sources = []
        for masked_splits_in_source in zip(*masked_splits):
            sources.append(masked_splits_in_source)

        masked_waveforms = [merge(x, padding_length) for x in sources]
        y = [
            utils.de_normalize_waveform(x, gain_factor, peak_gain_factor)
            in masked_waveforms
        ]
        return y
