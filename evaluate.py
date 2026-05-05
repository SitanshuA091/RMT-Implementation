import sacrebleu
import torch
import wandb


def generate_text(model, tokens, max_new_tokens, device):
    model.eval()
    tokens = tokens.to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def generate_text_memory(model, tokens, max_new_tokens, device):
    model.eval()
    tokens = tokens.to(device)
    memory = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, memory = model(tokens, memory)

            if memory is not None:
                memory = memory.detach()

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def compute_bleu(predictions, references):
    score = sacrebleu.corpus_bleu(predictions, references).score
    wandb.log({"eval/bleu": score})
    return score


def evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1)
            )

            total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    wandb.log({"eval/loss": avg_loss})

    return avg_loss


def evaluate_loss_memory(model, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            memory = None
            loss_sum = 0

            for s in range(batch.shape[1]):
                seg = batch[:, s, :]
                x, y = seg[:, :-1], seg[:, 1:]

                logits, memory = model(x, memory)

                if memory is not None:
                    memory = memory.detach()

                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                loss_sum += loss.item()

            total_loss += loss_sum

    avg_loss = total_loss / len(loader)

    wandb.log({"eval/loss": avg_loss})

    return avg_loss