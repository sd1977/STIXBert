"""Self-supervised pre-training pipeline for STIXBert."""

import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


def create_node_masks(
    data,
    mask_ratio: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Create random masks for masked node prediction.

    Args:
        data: PyG HeteroData.
        mask_ratio: Fraction of nodes to mask per type.

    Returns:
        Dict of node_type -> boolean mask tensor.
    """
    masks = {}
    for node_type in data.node_types:
        n = data[node_type].num_nodes
        num_mask = max(1, int(n * mask_ratio))
        mask = torch.zeros(n, dtype=torch.bool)
        indices = random.sample(range(n), num_mask)
        mask[indices] = True
        masks[node_type] = mask
    return masks


def sample_negative_edges(
    data,
    edge_type: tuple[str, str, str],
    num_neg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample negative edges (non-existent) for link prediction.

    Args:
        data: PyG HeteroData.
        edge_type: (src_type, rel_type, dst_type) tuple.
        num_neg: Number of negative edges to sample.

    Returns:
        Tuple of (src_indices, dst_indices) tensors.
    """
    src_type, _, dst_type = edge_type
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    # Existing edges as set for fast lookup
    existing = set()
    if edge_type in data.edge_types:
        ei = data[edge_type].edge_index
        for i in range(ei.shape[1]):
            existing.add((ei[0, i].item(), ei[1, i].item()))

    neg_src, neg_dst = [], []
    attempts = 0
    while len(neg_src) < num_neg and attempts < num_neg * 10:
        s = random.randint(0, num_src - 1)
        d = random.randint(0, num_dst - 1)
        if (s, d) not in existing:
            neg_src.append(s)
            neg_dst.append(d)
        attempts += 1

    return torch.tensor(neg_src), torch.tensor(neg_dst)


def sample_temporal_pairs(
    nodes: list[dict],
    num_pairs: int,
) -> list[tuple[int, int, float]]:
    """Sample pairs of nodes with temporal ordering labels.

    Args:
        nodes: List of STIX objects with timestamp fields.
        num_pairs: Number of pairs to sample.

    Returns:
        List of (idx_a, idx_b, label) where label=1 if a before b.
    """
    from datetime import datetime

    def parse_ts(obj):
        for field in ["first_seen", "created", "first_observed", "valid_from"]:
            ts = obj.get(field)
            if ts:
                try:
                    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    continue
        return None

    timestamped = [(i, parse_ts(n)) for i, n in enumerate(nodes)]
    timestamped = [(i, t) for i, t in timestamped if t is not None]

    if len(timestamped) < 2:
        return []

    pairs = []
    for _ in range(num_pairs):
        a, b = random.sample(timestamped, 2)
        label = 1.0 if a[1] < b[1] else 0.0
        pairs.append((a[0], b[0], label))

    return pairs


class PreTrainer:
    """Self-supervised pre-training loop for STIXBert."""

    def __init__(
        self,
        model,
        data,
        nodes_by_type: dict[str, list[dict]],
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        mask_ratio: float = 0.15,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.data = data
        self.nodes_by_type = nodes_by_type
        self.mask_ratio = mask_ratio
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Loss weights for the three objectives
        self.loss_weights = {
            "masked_node": 1.0,
            "link_pred": 1.0,
            "temporal": 0.5,
        }

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of loss component names to values.
        """
        self.model.train()

        # Move data to device
        x_dict = {
            nt: self.data[nt].x.to(self.device)
            for nt in self.data.node_types
        }
        edge_index_dict = {
            et: self.data[et].edge_index.to(self.device)
            for et in self.data.edge_types
        }

        # --- Masked Node Prediction ---
        masks = create_node_masks(self.data, self.mask_ratio)
        original_features = {nt: x.clone() for nt, x in x_dict.items()}

        # Zero out masked node features
        masked_x_dict = {}
        for nt, x in x_dict.items():
            x_masked = x.clone()
            if nt in masks:
                x_masked[masks[nt].to(self.device)] = 0.0
            masked_x_dict[nt] = x_masked

        # Forward pass with masked features
        embeddings = self.model(masked_x_dict, edge_index_dict)

        masks_device = {nt: m.to(self.device) for nt, m in masks.items()}
        loss_masked = self.model.masked_node_head(
            embeddings, original_features, masks_device
        )

        # --- Link Prediction ---
        loss_link = torch.tensor(0.0, device=self.device)
        link_count = 0
        for et in self.data.edge_types:
            ei = edge_index_dict[et]
            if ei.shape[1] == 0:
                continue

            src_type, _, dst_type = et
            pos_src_emb = embeddings[src_type][ei[0]]
            pos_dst_emb = embeddings[dst_type][ei[1]]

            num_neg = min(ei.shape[1], 1000)
            neg_src_idx, neg_dst_idx = sample_negative_edges(
                self.data, et, num_neg
            )
            neg_src_emb = embeddings[src_type][neg_src_idx.to(self.device)]
            neg_dst_emb = embeddings[dst_type][neg_dst_idx.to(self.device)]

            pos_logits = self.model.link_pred_head(pos_src_emb, pos_dst_emb)
            neg_logits = self.model.link_pred_head(neg_src_emb, neg_dst_emb)

            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([pos_labels, neg_labels])

            loss_link += nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            link_count += 1

        if link_count > 0:
            loss_link /= link_count

        # --- Temporal Ordering ---
        loss_temporal = torch.tensor(0.0, device=self.device)
        temporal_count = 0
        for nt, nodes in self.nodes_by_type.items():
            if nt not in embeddings:
                continue
            pairs = sample_temporal_pairs(nodes, num_pairs=min(len(nodes), 256))
            if not pairs:
                continue

            idx_a = torch.tensor([p[0] for p in pairs], device=self.device)
            idx_b = torch.tensor([p[1] for p in pairs], device=self.device)
            labels = torch.tensor(
                [p[2] for p in pairs], dtype=torch.float, device=self.device
            )

            emb_a = embeddings[nt][idx_a]
            emb_b = embeddings[nt][idx_b]
            logits = self.model.temporal_head(emb_a, emb_b)

            loss_temporal += nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            temporal_count += 1

        if temporal_count > 0:
            loss_temporal /= temporal_count

        # --- Combined Loss ---
        total_loss = (
            self.loss_weights["masked_node"] * loss_masked
            + self.loss_weights["link_pred"] * loss_link
            + self.loss_weights["temporal"] * loss_temporal
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "masked_node": loss_masked.item(),
            "link_pred": loss_link.item(),
            "temporal": loss_temporal.item(),
        }

    def train(
        self,
        num_epochs: int = 100,
        checkpoint_every: int = 10,
        log_every: int = 5,
    ) -> list[dict]:
        """Run full pre-training loop.

        Args:
            num_epochs: Number of training epochs.
            checkpoint_every: Save checkpoint every N epochs.
            log_every: Log losses every N epochs.

        Returns:
            List of per-epoch loss dicts.
        """
        history = []

        for epoch in range(1, num_epochs + 1):
            losses = self.train_epoch(epoch)
            history.append(losses)

            if epoch % log_every == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} | "
                    f"Total: {losses['total']:.4f} | "
                    f"Masked: {losses['masked_node']:.4f} | "
                    f"Link: {losses['link_pred']:.4f} | "
                    f"Temporal: {losses['temporal']:.4f}"
                )

            if epoch % checkpoint_every == 0:
                self.save_checkpoint(epoch)

        self.save_checkpoint(num_epochs, tag="final")
        return history

    def save_checkpoint(self, epoch: int, tag: Optional[str] = None) -> None:
        name = tag or f"epoch_{epoch}"
        path = self.checkpoint_dir / f"stixbert_{name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from epoch {epoch}: {path}")
        return epoch
