import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, random_split
from data import Tokenizer, ChatDataset
from model import TransformerLM
from training.optimizer import get_optimizer, get_scheduler
from training.trainer import Trainer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = Tokenizer()

    from model.config import ModelConfig
    import json

    with open("model_pretrained_config.json") as f:
        config = ModelConfig(**json.load(f))
    config.vocab_size = tokenizer.vocab_size

    model = TransformerLM(config)
    model.load_state_dict(torch.load("model_pretrained.pt", map_location=device, weights_only=True))
    print("Loaded pre-trained model")

    from datasets import load_dataset
    print("Loading ShareGPT dataset...")
    ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)

    conversations = []
    for i, example in enumerate(ds):
        if i >= 10000:
            break
        conv = []
        for turn in example.get("conversations", []):
            role = "user" if turn.get("from") == "human" else "assistant"
            conv.append({"role": role, "content": turn.get("value", "")})
        if conv:
            conversations.append(conv)

    print(f"Loaded {len(conversations)} conversations")

    dataset = ChatDataset(conversations, tokenizer, max_len=config.max_seq_len)
    val_size = min(len(dataset) // 20, 500)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 4
    grad_accum = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = get_optimizer(model, lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_scheduler(optimizer, warmup_steps=50, total_steps=total_steps)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        grad_accum_steps=grad_accum,
        device=device,
    )

    print("\nStarting SFT training...")
    trainer.train(num_epochs=3)

    torch.save(model.state_dict(), "model_chat.pt")
    import json
    with open("model_chat_config.json", "w") as f:
        json.dump(vars(config), f, indent=2)
    print("Chat model saved!")


if __name__ == "__main__":
    main()
