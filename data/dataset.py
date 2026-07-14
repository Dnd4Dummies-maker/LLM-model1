import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, tokens: list[int], seq_len: int = 1024):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1 : idx + 1 + self.seq_len], dtype=torch.long)
        return {"input_ids": x, "targets": y}


class ChatDataset(Dataset):
    def __init__(self, conversations: list[list[dict]], tokenizer, max_len: int = 1024):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for conv in conversations:
            tokens = []
            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    tokens.extend(tokenizer.encode(f"<|user|>\n{content}\n"))
                elif role == "assistant":
                    tokens.extend(tokenizer.encode(f"<|assistant|>\n{content}\n<|end|>\n"))

            for i in range(0, max(1, len(tokens) - max_len), max_len // 2):
                chunk = tokens[i : i + max_len]
                if len(chunk) > 1:
                    self.examples.append({
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "targets": torch.tensor(chunk[1:], dtype=torch.long),
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
