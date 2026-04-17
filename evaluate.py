import sacrebleu
import torch


def generate_text(model, tokens, max_new_tokens, device):
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(tokens)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

    return tokens


def compute_bleu(predictions, references):
    return sacrebleu.corpus_bleu(predictions, references).score