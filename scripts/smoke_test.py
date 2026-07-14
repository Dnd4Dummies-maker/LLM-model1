import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from model import TransformerLM, SMALL
from data import Tokenizer
from inference import generate

print("=== LLM Smoke Test ===\n")

# 1. Create tokenizer and config
tokenizer = Tokenizer()
config = SMALL
config.vocab_size = len(tokenizer)
print(f"Vocab size: {config.vocab_size}")

# 2. Build model
model = TransformerLM(config)
print(f"Parameters: {model.count_parameters():,}")

# 3. Test forward pass
x = torch.randint(0, config.vocab_size, (1, 32))
logits, loss = model(x, targets=x)
print(f"Forward pass OK: logits shape = {logits.shape}")

# 4. Test loss computation
targets = torch.randint(0, config.vocab_size, (1, 32))
_, loss = model(x, targets=targets)
print(f"Loss computation OK: loss = {loss.item():.4f}")

# 5. Test generation
prompt = "Hello, how are"
output = generate(model, prompt, tokenizer, max_new_tokens=20, device="cpu")
print(f"\nGeneration test:")
print(f"Prompt: {prompt}")
safe = output.encode("ascii", errors="replace").decode("ascii")
print(f"Output: {safe}")

# 6. Test training step
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
_, loss = model(x, targets=targets)
loss.backward()
optimizer.step()
print(f"\nTraining step OK: loss after step = {loss.item():.4f}")

# 7. Test save/load
torch.save(model.state_dict(), "test_model.pt")
model2 = TransformerLM(config)
model2.load_state_dict(torch.load("test_model.pt", weights_only=True))
print("Save/load OK")

import os
os.remove("test_model.pt")

print("\n=== All tests passed! ===")
