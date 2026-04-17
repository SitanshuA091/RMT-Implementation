import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import DecoderOnlyTransformer, MemoryDecoderTransformer
from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = GradScaler()
accumulation_steps = 8


def train_baseline(model, loader, optimizer):
    model.train()
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):

        x, y = x.to(device), y.to(device)

        with autocast():
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            ) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


def train_memory(model, loader, optimizer):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):

        batch = batch.to(device)
        memory = None

        with autocast():
            total_loss = 0

            for s in range(batch.shape[1]):

                seg = batch[:, s, :]
                x, y = seg[:, :-1], seg[:, 1:]

                logits, memory = model(x, memory)
                memory = memory.detach()

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += loss

            total_loss = total_loss / accumulation_steps

        scaler.scale(total_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()