import tiktoken


class Tokenizer:
    def __init__(self, name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)

    def __len__(self):
        return self.vocab_size
