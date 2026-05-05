import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb

from model import DecoderOnlyTransformer, MemoryDecoderTransformer
from dataset import *

device = "cuda" if torch.cuda.is_available() else "cpu"

scaler = GradScaler()
accumulation_steps = 8


def setup_wandb(config):
    wandb.init(
        project="rmt-implementation",
        config=config
    )


def train_baseline(model, loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()

    for step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        with autocast():
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            wandb.log({
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "step": step
            })


def train_memory(model, loader, optimizer, epoch):
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

                if memory is not None:
                    memory = memory.detach()

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                total_loss += loss

                if memory is not None:
                    wandb.log({
                        "memory/norm": memory.norm().item()
                    })

            scaled_loss = total_loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            wandb.log({
                "train/loss": total_loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/segments": batch.shape[1],
                "epoch": epoch,
                "step": step
            })


def main():
    config = {
        "lr": 3e-4,
        "batch_size": 32,
        "epochs": 10,
        "accumulation_steps": accumulation_steps,
        "model_type": "memory"
    }

    setup_wandb(config)

    dataset_raw = load_wikitext()
    train_texts = dataset_raw["train"]["text"]
    tokens = tokenize_texts(train_texts)

    if config["model_type"] == "baseline":
        dataset = LanguageModelDataset(tokens, block_size=128)
        model = DecoderOnlyTransformer()
    else:
        dataset = SegmentDataset(
            tokens,
            segment_length=128,
            segments_per_sample=4
        )
        model = MemoryDecoderTransformer()

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    model = model.to(device)

    wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    for epoch in range(config["epochs"]):
        if config["model_type"] == "baseline":
            train_baseline(model, loader, optimizer, epoch)
        else:
            train_memory(model, loader, optimizer, epoch)

        ckpt_path = f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)

        artifact = wandb.Artifact("rmt-model", type="model")
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)

        print(f"Epoch {epoch} completed")


if __name__ == "__main__":
    main()