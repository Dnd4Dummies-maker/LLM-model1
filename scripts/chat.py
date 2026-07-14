import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from data import Tokenizer
from model import TransformerLM
from inference import generate


def chat(model_path: str | None = None, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer()

    if model_path:
        import json
        from model.config import ModelConfig

        config_path = model_path.replace(".pt", "_config.json")
        with open(config_path) as f:
            config = ModelConfig(**json.load(f))
        config.vocab_size = len(tokenizer)
        model = TransformerLM(config)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded model from {model_path}")
    else:
        from model import SMALL
        config = SMALL
        config.vocab_size = len(tokenizer)
        model = TransformerLM(config)
        print("No model loaded - using random weights")

    model = model.to(device).eval()

    print("\nChat ready! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        response = generate(model, prompt, tokenizer, max_new_tokens=256, device=device)

        assistant_response = response.split("<|assistant|>\n")[-1].split("<|end|>")[0].strip()
        print(f"Assistant: {assistant_response}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with your LLM")
    parser.add_argument("model", nargs="?", default=None, help="Path to model .pt file")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    chat(args.model, args.device)
