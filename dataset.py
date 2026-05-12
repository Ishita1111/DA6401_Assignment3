from collections import Counter
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence

try:
    from datasets import load_dataset
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: 'datasets' (HuggingFace Datasets).\n"
        "Install it with:\n"
        "  python -m pip install -r requirements.txt\n"
        "If you're on Python 3.7, make sure 'requirements.txt' pins a compatible version."
    ) from e


def _load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as e:
        raise OSError(
            f"spaCy model '{model_name}' is not installed.\n"
            f"Install it with:\n"
            f"  python -m spacy download {model_name}"
        ) from e

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.dataset = load_dataset("bentrevett/multi30k")[split]

        self.spacy_de = _load_spacy_model("de_core_news_sm")
        self.spacy_en = _load_spacy_model("en_core_web_sm")

        self.special_tokens = {
            "<unk>": 0,
            "<pad>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

        if split == "train":
            self.build_vocab()

        self.data = self.process_data()

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.dataset:
            de_tokens = [
                token.text.lower()
                for token in self.spacy_de(example["de"])
            ]

            en_tokens = [
                token.text.lower()
                for token in self.spacy_en(example["en"])
            ]

            src_counter.update(de_tokens)
            tgt_counter.update(en_tokens)

        self.src_vocab = dict(self.special_tokens)
        self.tgt_vocab = dict(self.special_tokens)

        for token in src_counter:
            if token not in self.src_vocab:
                self.src_vocab[token] = len(self.src_vocab)

        for token in tgt_counter:
            if token not in self.tgt_vocab:
                self.tgt_vocab[token] = len(self.tgt_vocab)

        self.src_itos = {idx: tok for tok, idx in self.src_vocab.items()}
        self.tgt_itos = {idx: tok for tok, idx in self.tgt_vocab.items()}

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        processed_data = []

        for example in self.dataset:

            de_tokens = [
                token.text.lower()
                for token in self.spacy_de(example["de"])
            ]

            en_tokens = [
                token.text.lower()
                for token in self.spacy_en(example["en"])
            ]

            src_indices = [self.src_vocab["<sos>"]]

            src_indices += [
                self.src_vocab.get(token, self.src_vocab["<unk>"])
                for token in de_tokens
            ]

            src_indices.append(self.src_vocab["<eos>"])

            tgt_indices = [self.tgt_vocab["<sos>"]]

            tgt_indices += [
                self.tgt_vocab.get(token, self.tgt_vocab["<unk>"])
                for token in en_tokens
            ]

            tgt_indices.append(self.tgt_vocab["<eos>"])

            processed_data.append(
                {
                    "src": torch.tensor(src_indices, dtype=torch.long),
                    "tgt": torch.tensor(tgt_indices, dtype=torch.long),
                }
            )

        return processed_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def collate_fn(batch):

    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=1
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=1
    )

    return {
        "src": src_batch,
        "tgt": tgt_batch
    }