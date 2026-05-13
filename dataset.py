from datasets import load_dataset
from collections import Counter

import spacy
import torch

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class Multi30kDataset:

    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """

        print(f"\nLoading split: {split}")

        self.split = split

        # Load dataset
        self.dataset = load_dataset(
            "bentrevett/multi30k"
        )[split]

        print(f"{split} dataset loaded")
        print(f"Number of samples: {len(self.dataset)}")

        # Load lightweight tokenizer-only pipelines
        print("Loading spacy tokenizers...")

        self.spacy_de = spacy.load(
            "de_core_news_sm",
            disable=["parser", "tagger", "ner", "lemmatizer"]
        )

        self.spacy_en = spacy.load(
            "en_core_web_sm",
            disable=["parser", "tagger", "ner", "lemmatizer"]
        )

        print("Spacy tokenizers loaded")

        self.special_tokens = {
            "<unk>": 0,
            "<pad>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

        # Build vocab only on train split
        if split == "train":

            print("\nBuilding vocabulary...")
            self.build_vocab()
            print("Vocabulary built")

        print("\nProcessing dataset...")
        if split == "train":
            self.data = self.process_data()
        else:
            self.data = []
        print("Dataset processing complete")

    # ─────────────────────────────────────────────
    # BUILD VOCAB
    # ─────────────────────────────────────────────

    def build_vocab(self):

        src_counter = Counter()
        tgt_counter = Counter()

        progress_bar = tqdm(
            self.dataset,
            desc="Building vocab"
        )

        for example in progress_bar:

            de_tokens = [
                token.text.lower()
                for token in self.spacy_de.tokenizer(example["de"])
            ]

            en_tokens = [
                token.text.lower()
                for token in self.spacy_en.tokenizer(example["en"])
            ]

            src_counter.update(de_tokens)
            tgt_counter.update(en_tokens)

        print(f"German vocab size before specials: {len(src_counter)}")
        print(f"English vocab size before specials: {len(tgt_counter)}")

        self.src_vocab = dict(self.special_tokens)
        self.tgt_vocab = dict(self.special_tokens)

        # Build src vocab
        for token in src_counter:

            if token not in self.src_vocab:
                self.src_vocab[token] = len(self.src_vocab)

        # Build tgt vocab
        for token in tgt_counter:

            if token not in self.tgt_vocab:
                self.tgt_vocab[token] = len(self.tgt_vocab)

        self.src_itos = {
            idx: tok
            for tok, idx in self.src_vocab.items()
        }

        self.tgt_itos = {
            idx: tok
            for tok, idx in self.tgt_vocab.items()
        }

        print(f"Final German vocab size: {len(self.src_vocab)}")
        print(f"Final English vocab size: {len(self.tgt_vocab)}")

    # ─────────────────────────────────────────────
    # PROCESS DATA
    # ─────────────────────────────────────────────

    def process_data(self):

        processed_data = []

        progress_bar = tqdm(
            self.dataset,
            desc=f"Tokenizing {self.split}"
        )

        for example in progress_bar:

            de_tokens = [
                token.text.lower()
                for token in self.spacy_de.tokenizer(example["de"])
            ]

            en_tokens = [
                token.text.lower()
                for token in self.spacy_en.tokenizer(example["en"])
            ]

            src_indices = [self.src_vocab["<sos>"]]

            src_indices += [
                self.src_vocab.get(
                    token,
                    self.src_vocab["<unk>"]
                )
                for token in de_tokens
            ]

            src_indices.append(
                self.src_vocab["<eos>"]
            )

            tgt_indices = [self.tgt_vocab["<sos>"]]

            tgt_indices += [
                self.tgt_vocab.get(
                    token,
                    self.tgt_vocab["<unk>"]
                )
                for token in en_tokens
            ]

            tgt_indices.append(
                self.tgt_vocab["<eos>"]
            )

            processed_data.append({
                "src": torch.tensor(
                    src_indices,
                    dtype=torch.long
                ),
                "tgt": torch.tensor(
                    tgt_indices,
                    dtype=torch.long
                ),
            })

        return processed_data

    # ─────────────────────────────────────────────
    # DATASET METHODS
    # ─────────────────────────────────────────────

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────────────────────────
# COLLATE FUNCTION
# ─────────────────────────────────────────────

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