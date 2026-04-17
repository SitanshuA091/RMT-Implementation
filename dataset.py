from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def load_wikitext():
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    return dataset


def tokenize_texts(texts):
    tokens = []
    for t in texts:
        if len(t.strip()) == 0:
            continue
        tokens.extend(tokenizer.encode(t))
    return tokens


class LanguageModelDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1])
        y = torch.tensor(chunk[1:])
        return x, y


class SegmentDataset(Dataset):
    def __init__(self, tokens, segment_length, segments_per_sample):
        self.tokens = tokens
        self.segment_length = segment_length
        self.segments_per_sample = segments_per_sample

        total = segment_length * segments_per_sample
        self.samples = len(tokens) // total

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        start = idx * self.segment_length * self.segments_per_sample

        segments = []
        for i in range(self.segments_per_sample):
            seg = self.tokens[
                start + i*self.segment_length :
                start + (i+1)*self.segment_length
            ]
            segments.append(torch.tensor(seg))

        return torch.stack(segments)