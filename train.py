"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

import evaluate
from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        smoothing: float = 0.1,
    ) -> None:

        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():

            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(
                self.smoothing / (self.vocab_size - 2)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence
            )

            true_dist[:, self.pad_idx] = 0

            pad_mask = (target == self.pad_idx)

            true_dist[pad_mask] = 0

        loss = torch.sum(-true_dist * log_probs, dim=1)

        non_pad_mask = (target != self.pad_idx)

        loss = loss.masked_select(non_pad_mask).mean()

        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for batch in data_iter:

        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_input).to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask
            )

            logits = logits.reshape(-1, logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = loss_fn(logits, tgt_output)

            if is_train:
                loss.backward()

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_iter)

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:

    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    for _ in range(max_len - 1):

        tgt_mask = make_tgt_mask(ys)

        out = model.decode(
            memory,
            src_mask,
            ys,
            tgt_mask
        )

        prob = out[:, -1]

        next_word = torch.argmax(prob, dim=-1).item()

        next_word_tensor = torch.ones(1, 1).type_as(src).fill_(next_word)

        ys = torch.cat(
            [ys, next_word_tensor],
            dim=1
        )

        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:

    model.eval()

    bleu_metric = evaluate.load("bleu")

    predictions = []
    references = []

    sos_idx = 2
    eos_idx = 3
    pad_idx = 1

    with torch.no_grad():

        for batch in test_dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            for i in range(src.size(0)):

                src_sentence = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_sentence).to(device)

                pred_tokens = greedy_decode(
                    model,
                    src_sentence,
                    src_mask,
                    max_len,
                    sos_idx,
                    eos_idx,
                    device
                )

                pred_tokens = pred_tokens.squeeze(0).tolist()
                tgt_tokens = tgt[i].tolist()

                pred_sentence = []
                tgt_sentence = []

                for idx in pred_tokens:

                    if idx in [sos_idx, eos_idx, pad_idx]:
                        continue

                    if hasattr(tgt_vocab, "itos"):
                        pred_sentence.append(tgt_vocab.itos[idx])
                    else:
                        pred_sentence.append(
                            tgt_vocab.lookup_token(idx)
                        )

                for idx in tgt_tokens:

                    if idx in [sos_idx, eos_idx, pad_idx]:
                        continue

                    if hasattr(tgt_vocab, "itos"):
                        tgt_sentence.append(tgt_vocab.itos[idx])
                    else:
                        tgt_sentence.append(
                            tgt_vocab.lookup_token(idx)
                        )

                predictions.append(pred_sentence)
                references.append([tgt_sentence])

    bleu_score = bleu_metric.compute(
        predictions=predictions,
        references=references
    )

    return bleu_score["bleu"] * 100


# ══════════════════════════════════════════════════════════════════════
#   CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # "model_config": {
        #     "src_vocab_size": model.src_embedding.num_embeddings,
        #     "tgt_vocab_size": model.tgt_embedding.num_embeddings,
        #     "d_model": model.d_model,
        # } 
        "model_config": model.config
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:

    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    if scheduler is not None:
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:

    print("Training pipeline setup complete.")
    print("Implement dataset + dataloaders before running training.")


if __name__ == "__main__":
    run_training_experiment()