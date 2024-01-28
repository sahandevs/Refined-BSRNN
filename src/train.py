from .utils import sep, GLOBAL_CONFIG
from .model import ModelConfig, init_model_arg_parser

from argparse import _SubParsersAction, ArgumentParser
import utils
from pprint import pprint
import random
from pathlib import Path
import os
import math
import datetime

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.functional as F
import torchaudio.transforms as T
import numpy as np

from ignite.engine import (
    Engine,
    Events,
)
from ignite.metrics import RunningAverage
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    global_step_from_engine,
)
from ignite.contrib.handlers import (
    TensorboardLogger,
    global_step_from_engine,
    TensorboardLogger,
)
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Timer


def init_arg_parser(subparsers: _SubParsersAction[ArgumentParser]):
    COMMAND_NAME = "train"
    parser = subparsers.add_parser(COMMAND_NAME, help="train mode")
    init_model_arg_parser(parser)
    parser.add_argument(
        "--portion",
        type=float,
        default=1.0,
        help="percentage (0.0 to 1.0) of dataset to use. Useful for experimenting with smaller dataset",
    )
    parser.add_argument("--clip_grad_norm", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--data_loader_workers", type=int, default=1)

    parser.add_argument(
        "--musdbhq_location",
        type=str,
        required=True,
        help="location of downloaded dataset (location of the root folder. the root folder should contain two sub-folders named 'train' and 'test')",
    )
    parser.add_argument(
        "--checkpoint_fp",
        type=str,
        required=False,
        help="location of checkpoint file (.pt)",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="unique name for run name",
    )

    return COMMAND_NAME, parser


class Config(ModelConfig):
    clip_grad_norm: int
    max_epochs: int
    lr: int
    data_loader_workers: int
    batch_size: int
    data_loader_workers: int
    portion: float
    musdbhq_location: str
    name: str
    checkpoint_fp: str

    def __init__(self, args: dict):
        super().__init__(args)
        self.batch_size = args["--batch_size"]
        self.data_loader_workers = args["--data_loader_workers"]
        self.portion = args["--portion"]
        self.musdbhq_location = args["--musdbhq_location"]
        self.checkpoint_fp = args["--checkpoint_fp"] or ""
        self.name = args["--name"]
        self.clip_grad_norm = args["--clip_grad_norm"]
        self.max_epochs = args["--max_epochs"]
        self.lr = args["--lr"]


class CustomLoss(nn.Module):
    """BSRNN custom loss implementation"""

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mae_stft_real = nn.L1Loss()
        self.mae_stft_imag = nn.L1Loss()
        self.mae_inv_stft = nn.L1Loss()

    def forward(self, pred_stft, target_stft, pred_inv_stft, target_inv_stft):
        loss_r = self.mae_stft_real(pred_stft.real, target_stft.real)
        loss_i = self.mae_stft_imag(pred_stft.imag, target_stft.imag)
        loss_t = self.mae_inv_stft(pred_inv_stft, target_inv_stft)
        loss = loss_r + loss_i + loss_t
        return loss


def compute_usdr(pred, target, delta=1e-7):
    """uSDR evaluation metric"""
    if pred.shape[0] < target.shape[0]:
        padding = target.shape[0] - pred.shape[0]
        pred = torch.nn.functional.pad(pred, (0, padding), "constant", 0)
    num = torch.sum(torch.square(target))
    den = torch.sum(torch.square(target - pred))
    num += delta
    den += delta
    usdr = 10 * torch.log10(num / den)
    return usdr.mean()


all_parts = ["bass", "drums", "other", "vocals"]


class Dataset(torch.utils.data.IterableDataset):
    """MusDB18-hq dataset loader"""

    def __init__(self, cfg: Config, parts=all_parts, validation=False):
        super(Dataset, self).__init__()
        path = Path(cfg.musdbhq_location) / ("test" if validation else "train")
        self.parts = parts
        self.cfg = cfg
        self.files = [entry for entry in os.scandir(path) if entry.is_dir()]
        # we are seeding the random gen manually so this is fine.
        random.shuffle(self.files)
        self.files = self.files[: int(len(self.files) * cfg.portion)]
        self.loaded = {}
        torch.set_default_device("cpu")
        self.to_stft = utils.to_spectrogram(cfg)
        print("pre-loading")
        targets = [f"{d.path}/{part}.wav" for part in self.parts for d in self.files]
        [targets.append(f"{d.path}/mixture.wav") for d in self.files]
        from tqdm import tqdm

        self.cache = True
        for x in tqdm(targets[: int(len(targets) * 0.2)]):
            self.load_audio(x)
        self.cache = False
        print("pre-loading done")

        torch.set_default_device(GLOBAL_CONFIG.device)

    def load_audio(self, path):
        if path in self.loaded:
            return self.loaded[path]
        x = utils.load_audio(path, self.cfg)
        x, _, _ = utils.normalize_waveform(x, self.cfg)
        x, _ = utils.split(x, cfg=self.cfg, drop_last=True)
        x = torch.stack([self.to_stft(k) for k in x], dim=0)
        if self.cache:
            self.loaded[path] = x
        return x

    def iterator(self):
        torch.set_default_device("cpu")

        for i in range(self.start_index, self.end_index):
            d = self.files[i]

            mixture = f"{d.path}/mixture.wav"
            targets = [f"{d.path}/{part}.wav" for part in self.parts]

            mix_stft_splits = self.load_audio(mixture)
            target_stfts_splits = [self.load_audio(target) for target in targets]

            for i, mix_stft in enumerate(mix_stft_splits):
                target_stfts = [x[i] for x in target_stfts_splits]
                out = (d.path, mix_stft, target_stfts)
                yield out

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            total_files = len(self.files)
            per_worker = int(math.ceil(total_files / float(worker_info.num_workers)))
            self.start_index = worker_info.id * per_worker
            self.end_index = min(self.start_index + per_worker, total_files)
        else:
            self.start_index = 0
            self.end_index = len(self.files)

        return iter(self.iterator())


def collate_fn(batch):
    # Dataloader returns the following shape:
    # (d.path, mix_stft (1025, 87), target_stfts (4, 1025, 87))
    # we have to add the batch dim to it

    b_mix_stft = []
    b_target_stfts = [[] for _ in range(len(batch[0][2]))]

    for x, mix, target in batch:
        b_mix_stft.append(mix)
        for i, t in enumerate(target):
            b_target_stfts[i].append(t)
    # (B, 1025, 87), (4, B, 1025, 87)
    return torch.stack(b_mix_stft), [torch.stack(x) for x in b_target_stfts]


def do_train(parser: ArgumentParser):
    cfg = Config(parser.parse_args())

    print("Train mode Configuration:")
    pprint(cfg.__dict__)
    sep()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # dataloader opens too many files
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    parts = all_parts  # TODO:
    t_dataset = Dataset(validation=False, parts=parts, cfg=cfg)
    v_dataset = Dataset(validation=True, parts=parts, cfg=cfg)

    import gc

    torch.cuda.empty_cache()
    gc.collect()

    train_loader = torch.utils.data.DataLoader(
        t_dataset,
        num_workers=cfg.data_loader_workers,
        batch_size=cfg.batch_size,
        drop_last=True,
        prefetch_factor=4,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    val_loader = torch.utils.data.DataLoader(
        v_dataset,
        num_workers=cfg.data_loader_workers,
        batch_size=cfg.batch_size,
        drop_last=True,
        prefetch_factor=4,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model = cfg.create_model().to(GLOBAL_CONFIG.device)
    fg_prefix = "generic-split" if cfg.generic_bands else "v7-split"
    name_prefix = (
        "base-model"
        if cfg.checkpoint_fp == ""
        else f"base-model-to-{','.join(cfg.parts)}"
    )

    name = f"{fg_prefix}-{name_prefix}-portion-{cfg.portion}"
    prefix = f"./train-logs/{name}-{datetime.datetime.now()}"
    if not os.path.exists(name):
        os.makedirs(prefix)
    print(f"run name={prefix}\nparams={utils.count_model_parameters(model=model)}\n")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # trainer setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = CustomLoss()
    torch.set_default_device(GLOBAL_CONFIG.device)
    inv_stft_gpu = utils.from_spectrogram(cfg)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        mix_stft, mask_stfts = batch
        mix_stft, mask_stfts = mix_stft.to(GLOBAL_CONFIG.device), mask_stfts
        mask_stfts = [x.to(GLOBAL_CONFIG.device) for x in mask_stfts]

        y_masks = model(mix_stft)  # (num_sources, B, 1025, 87)
        inv_y_masks = [inv_stft_gpu(y_mask) for y_mask in y_masks]
        # (num_sources, B, 1025, 87)
        inv_mask_stfts = [inv_stft_gpu(mask_stft) for mask_stft in mask_stfts]

        total_loss = 0
        for y_mask, mask_stft, inv_y_mask, inv_mask_stft in zip(
            y_masks, mask_stfts, inv_y_masks, inv_mask_stfts
        ):
            total_loss += loss_fn(y_mask, mask_stft, inv_y_mask, inv_mask_stft)
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        optimizer.step()

        # evaluate
        usdr_values = [
            compute_usdr(inv_y_mask, inv_mask_stft).item()
            for inv_y_mask, inv_mask_stft in zip(inv_y_masks, inv_mask_stfts)
        ]
        return {"loss": total_loss.item(), "usdr": np.mean(usdr_values)}

    trainer = Engine(train_step)

    # Metrics
    running_avg_usdr = RunningAverage(output_transform=lambda x: x["usdr"])
    running_avg_usdr.attach(trainer, "running_avg_usdr")
    running_avg_loss = RunningAverage(output_transform=lambda x: x["loss"])
    running_avg_loss.attach(trainer, "running_avg_loss")

    tb_logger = TensorboardLogger(log_dir=f"{prefix}/tb")
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda x: {
            "running_avg_loss": trainer.state.metrics["running_avg_loss"],
            "running_avg_usdr": trainer.state.metrics["running_avg_usdr"],
        },
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # evaluator setup
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def eval_step(engine, batch):
        model.eval()
        with torch.no_grad():
            mix_stft, mask_stfts = batch
            mix_stft, mask_stfts = mix_stft.to("cuda"), mask_stfts
            mask_stfts = [mask_stft.to("cuda") for mask_stft in mask_stfts]

            y_masks = model(mix_stft)
            inv_y_masks = [inv_stft_gpu(y_mask) for y_mask in y_masks]
            inv_mask_stfts = [inv_stft_gpu(mask_stft) for mask_stft in mask_stfts]

            usdr_values = [
                compute_usdr(inv_y_mask, inv_mask_stft).item()
                for inv_y_mask, inv_mask_stft in zip(inv_y_masks, inv_mask_stfts)
            ]

            return {"usdr": np.mean(usdr_values)}

    evaluator = Engine(eval_step)
    running_avg_usdr_eval = RunningAverage(output_transform=lambda x: x["usdr"])
    running_avg_usdr_eval.attach(evaluator, "running_avg_usdr_eval")

    # reporting

    iter_timer = Timer(average=True)
    epoch_timer = Timer(average=False)

    @evaluator.on(Events.COMPLETED)
    def log_results(engine):
        usdr = engine.state.metrics["running_avg_usdr_eval"]
        print(
            f"Test Results - Avg usdr: {usdr} - iter_timer:{iter_timer.value()}s - epoch_timer:{epoch_timer.value()}"
        )

    bar = ProgressBar(persist=False)
    bar.attach(evaluator, metric_names=["running_avg_usdr_eval"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def train_epoch_completed(engine):
        epoch = engine.state.epoch
        evaluator.run(val_loader)
        tb_logger.add_scalar(
            "evaluation/running_avg_usdr_eval",
            evaluator.state.metrics["running_avg_usdr_eval"],
            epoch,
        )

    iter_timer.attach(
        trainer,
        start=Events.STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    epoch_timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        step=Events.EPOCH_COMPLETED,
    )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model checkpointing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "running_avg_usdr": running_avg_usdr,
        "running_avg_loss": running_avg_loss,
        "cfg": cfg,
    }
    checkpoint_handler = Checkpoint(
        to_save,
        DiskSaver(f"{prefix}/models", create_dir=True),
        n_saved=25,
        global_step_transform=global_step_from_engine(trainer),
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)

    if cfg.checkpoint_fp and os.path.isfile(cfg.checkpoint_fp):
        checkpoint = torch.load(cfg.checkpoint_fp)
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)
    else:
        print(f"{cfg.checkpoint_fp} not found")
        raise ""

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # setup transfer learning
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if False:  # TODO:
        new_model = BSRNN(num_sources=1).to("cuda")
        print(f"Params checkpoint -> {count_parameters(model)}")
        new_model.split = model.split
        new_model.sequence = model.sequence
        new_model.mask = nn.ModuleList(
            [
                MaskEstimation(
                    band_indices=new_model.split.band_indices,
                    fully_connected_out=new_model.split.fully_connected_out,
                )
            ]
        )
        model = new_model
        print(f"Params before freeze -> {count_parameters(model)}")
        # Freeze first two modules. keep the mask estimation
        for param in model.split.parameters():
            param.requires_grad = False
        for param in model.sequence.parameters():
            param.requires_grad = False
        for param in model.masks[0].parameters():
            param.requires_grad = True
        running_avg_usdr.reset()
        running_avg_loss.reset()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer.state.max_epochs = None
        print(f"Params after freeze -> {count_parameters(model)}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # start training
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=["running_avg_loss", "running_avg_usdr"])
    trainer.run(train_loader, max_epochs=cfg.max_epochs)
    print("Done! saved in {prefix}")
