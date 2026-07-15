import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inference import generate


def chat(model_path: str | None = None, device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_path and Path(model_path).is_dir():
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        model = model.to(device).eval()
        print(f"Loaded HuggingFace model from {model_path}")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(output[0], skip_special_tokens=False)
            reply = response.split("<|assistant|>\n")[-1].split("<|end|>")[0].strip()
            print(f"Assistant: {reply}\n")

    elif model_path and model_path.endswith(".pt"):
        from data import Tokenizer
        from model import TransformerLM
        from model.config import ModelConfig
        import json

        tokenizer = Tokenizer()
        config_path = model_path.replace(".pt", "_config.json")
        with open(config_path) as f:
            config = ModelConfig(**json.load(f))
        config.vocab_size = len(tokenizer)
        model = TransformerLM(config)

        state = torch.load(model_path, map_location=device, weights_only=True)
        remap = {
            "tok_emb.": "token_embedding.",
            "head.": "lm_head.",
            ".attn.": ".attention.",
        }
        new_state = {}
        for k, v in state.items():
            for old, new in remap.items():
                k = k.replace(old, new)
            new_state[k] = v
        model.load_state_dict(new_state)
        model = model.to(device).eval()
        print(f"Loaded model from {model_path}")

        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            response = generate(model, prompt, tokenizer, max_new_tokens=256, device=device)
            assistant_response = response.split("<|assistant|>\n")[-1].split("<|end|>")[0].strip()
            print(f"Assistant: {assistant_response}\n")
    else:
        print("Usage:")
        print("  python chat.py chat_gpt2           (HuggingFace format)")
        print("  python chat.py model_chat.pt       (custom .pt format)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chat with your LLM")
    parser.add_argument("model", nargs="?", default=None, help="Path to model dir or .pt file")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()
    chat(args.model, args.device)
