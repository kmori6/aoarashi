import re
import unicodedata

from g2p_en import G2p
from sentencepiece import SentencePieceProcessor


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    # remove all punctuation marks except five (',?.!) following https://arxiv.org/abs/2406.19674.
    text = re.sub(r"[^\w\s',?.!]", "", text)
    # remove all duplicate whitespaces
    text = re.sub(r"\s+", " ", text)
    return text


class Tokenizer:
    def __init__(self, model_path: str):
        self.sp_model = SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> list:
        return self.sp_model.encode(text)

    def decode(self, ids: list) -> str:
        return self.sp_model.decode(ids)


class PhonemeTokenizer:
    def __init__(self, token_path: str):
        with open(token_path, "r", encoding="utf-8") as f:
            self.token_list = [line.strip() for line in f.readlines()]
        self.token2idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.g2p = G2p()

    def encode(self, text: str) -> list:
        phonemes = [p for p in self.g2p(text.strip()) if p != " "]
        return [self.token2idx.get(p, self.token2idx["<unk>"]) for p in phonemes]

    def decode(self, ids: list) -> str:
        return "".join([self.token_list[i] for i in ids])
