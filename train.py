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

from nltk.translate.bleu_score import corpus_bleu

from collections import Counter
import math
from model import Transformer, make_src_mask, make_tgt_mask
import tqdm


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

print("Starting training pipeline...", flush=True)

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

    import wandb
    from tqdm import tqdm

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    progress_bar = tqdm(
        data_iter,
        desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num+1}",
        leave=False
    )

    for batch_idx, batch in enumerate(progress_bar):

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

            vocab_size = logits.size(-1)

            logits_flat = logits.reshape(-1, vocab_size)
            tgt_output_flat = tgt_output.reshape(-1)

            loss = loss_fn(
                logits_flat,
                tgt_output_flat
            )
            
            accuracy = compute_accuracy(
                logits,
                tgt_output
            )

            total_accuracy += accuracy

            # ─────────────────────────────────────
            # Prediction confidence
            # ─────────────────────────────────────

            probs = torch.softmax(
                logits_flat,
                dim=-1
            )

            max_probs = probs.max(dim=-1)[0]

            avg_confidence = max_probs.mean().item()

            if is_train:

                loss.backward()

                # ─────────────────────────────────
                # Gradient norm
                # ─────────────────────────────────

                total_norm = 0.0

                for p in model.parameters():

                    if p.grad is not None:

                        param_norm = p.grad.data.norm(2)

                        total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5

                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # ─────────────────────────────────
                # WandB batch logging
                # ─────────────────────────────────

                wandb.log({
                    "batch_loss": loss.item(),
                    "gradient_norm": total_norm,
                    "prediction_confidence": avg_confidence,
                    "learning_rate": optimizer.param_groups[0]["lr"]
                })

        total_loss += loss.item()

        avg_loss_so_far = total_loss / (batch_idx + 1)

        avg_accuracy_so_far = total_accuracy / (batch_idx + 1)

        progress_bar.set_postfix({
            "loss": f"{avg_loss_so_far:.4f}",
            "acc": f"{avg_accuracy_so_far:.4f}"
        })

    avg_loss = total_loss / len(data_iter)
    avg_accuracy = total_accuracy / len(data_iter)

    return avg_loss, avg_accuracy

def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_idx: int = 1
):
    predictions = torch.argmax(
        logits,
        dim=-1
    )

    mask = targets != pad_idx
    correct = (
        (predictions == targets) & mask
    ).sum().item()

    total = mask.sum().item()
    return correct / total


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

    hypotheses = []
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

                    pred_sentence.append(
                        tgt_vocab.tgt_itos.get(idx, "<unk>")
                    )

                for idx in tgt_tokens:

                    if idx in [sos_idx, eos_idx, pad_idx]:
                        continue

                    tgt_sentence.append(
                        tgt_vocab.tgt_itos.get(idx, "<unk>")
                    )

                hypotheses.append(pred_sentence)
                references.append([tgt_sentence])

    bleu_score = corpus_bleu(
        references,
        hypotheses
    ) * 100

    return bleu_score


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
        "src_vocab": model.src_vocab,
        "tgt_vocab": model.tgt_vocab,
        "src_itos": model.src_itos,
        "tgt_itos": model.tgt_itos,
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
    
    if "src_vocab" in checkpoint:
        model.src_vocab = checkpoint["src_vocab"]

    if "tgt_vocab" in checkpoint:
        model.tgt_vocab = checkpoint["tgt_vocab"]

    if "tgt_itos" in checkpoint:
        model.tgt_itos = checkpoint["tgt_itos"]

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
    
    print("Starting datasets...", flush=True)

    import wandb

    from dataset import Multi30kDataset, collate_fn
    from lr_scheduler import NoamScheduler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="da6401-Assignment3",
        name="noam_scheduler_run",
        group="2.1",
        config={
            "batch_size": 128,
            "num_epochs": 10,
            "d_model": 512,
            "num_layers": 4,
            "num_heads": 8,
            "d_ff": 2048,
            "dropout": 0.05,
            "warmup_steps": 8000,
            "learning_rate": 1.0,
            "label_smoothing": 0.1,
        }
    )

    config = wandb.config

    # ─────────────────────────────────────────────
    # DATASETS
    # ─────────────────────────────────────────────

    print("Loading training dataset...", flush=True)

    train_dataset = Multi30kDataset(split="train")

    print("Loading validation dataset...", flush=True)

    val_dataset = Multi30kDataset(split="validation")

    val_dataset.src_vocab = train_dataset.src_vocab
    val_dataset.tgt_vocab = train_dataset.tgt_vocab
    val_dataset.src_itos = train_dataset.src_itos
    val_dataset.tgt_itos = train_dataset.tgt_itos
    
    print("Processing validation dataset...", flush=True)
    
    val_dataset.data = val_dataset.process_data()

    print("Validation dataset processed", flush=True)

    print("Loading test dataset...", flush=True)

    test_dataset = Multi30kDataset(split="test")

    test_dataset.src_vocab = train_dataset.src_vocab
    test_dataset.tgt_vocab = train_dataset.tgt_vocab
    test_dataset.src_itos = train_dataset.src_itos
    test_dataset.tgt_itos = train_dataset.tgt_itos
    
    print("Processing test dataset...", flush=True)
    
    test_dataset.data = test_dataset.process_data()

    print("Test dataset processed", flush=True)

    # ─────────────────────────────────────────────
    # DATALOADERS
    # ─────────────────────────────────────────────

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ─────────────────────────────────────────────
    # MODEL
    # ─────────────────────────────────────────────

    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config.d_model,
        N=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        checkpoint_path=None
    ).to(device)

    # required for infer() and evaluate_bleu()
    model.src_vocab = train_dataset.src_vocab
    model.tgt_vocab = train_dataset.tgt_vocab
    
    model.src_itos = train_dataset.src_itos
    model.tgt_itos = train_dataset.tgt_itos

    # ─────────────────────────────────────────────
    # OPTIMIZER
    # ─────────────────────────────────────────────

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=config.d_model,
        warmup_steps=config.warmup_steps
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=1,
        smoothing=config.label_smoothing
    )

    # ─────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────

    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):

        train_loss, train_acc = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device
        )

        val_loss, val_acc = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"]
        })

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path="best_checkpoint.pt"
            )

    # ─────────────────────────────────────────────
    # FINAL BLEU
    # ─────────────────────────────────────────────

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_dataset,
        device=device
    )

    print(f"Test BLEU: {bleu:.2f}")

    wandb.log({
        "test_bleu": bleu
    })

    wandb.finish()

if __name__ == "__main__":
    run_training_experiment()