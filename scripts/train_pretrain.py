import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from data import Tokenizer, PretrainDataset, ChatDataset
from model import TransformerLM, SMALL
from training.optimizer import get_optimizer, get_scheduler
from training.trainer import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config = SMALL
    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Estimated parameters: {config.parameter_count_estimate() / 1e6:.1f}M")

    model = TransformerLM(config)

    from datasets import load_dataset
    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)

    print("Tokenizing...")
    all_tokens = []
    for i, example in enumerate(ds):
        if i >= 20000:
            break
        text = example.get("instruction", "") + "\n" + example.get("input", "") + "\n" + example.get("output", "")
        all_tokens.extend(tokenizer.encode(text))

    print(f"Total tokens: {len(all_tokens):,}")

    seq_len = config.max_seq_len
    dataset = PretrainDataset(all_tokens, seq_len)
    val_size = min(len(dataset) // 20, 1000)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 8
    grad_accum = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = get_optimizer(model, lr=3e-4)
    total_steps = len(train_loader) * 5
    scheduler = get_scheduler(optimizer, warmup_steps=100, total_steps=total_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_accum_steps=grad_accum,
        device=device,
    )

    print("\nStarting pre-training...")
    trainer.train(num_epochs=5)

    torch.save(model.state_dict(), "model_pretrained.pt")
    import json
    with open("model_pretrained_config.json", "w") as f:
        json.dump({k: v for k, v in vars(config).items()}, f, indent=2)
    print("Model saved!")


if __name__ == "__main__":
    main()
