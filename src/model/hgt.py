"""STIXBert: HGT-based model for self-supervised pre-training on STIX graphs."""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear


class STIXBertEncoder(nn.Module):
    """Heterogeneous Graph Transformer encoder for STIX graphs.

    Implements type-aware attention over heterogeneous STIX object graphs
    following Hu et al. (2020) HGT architecture.
    """

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_types = node_types
        self.edge_types = edge_types
        self.num_layers = num_layers

        # Per-type input projection
        self.input_projections = nn.ModuleDict({
            nt: Linear(input_dim, hidden_dim)
            for nt in node_types
        })

        # HGT convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=(node_types, edge_types),
                    heads=num_heads,
                )
            )
            self.norms.append(
                nn.ModuleDict({
                    nt: nn.LayerNorm(hidden_dim) for nt in node_types
                })
            )

        # Output projection
        self.output_projection = nn.ModuleDict({
            nt: Linear(hidden_dim, output_dim)
            for nt in node_types
        })

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        """Forward pass through HGT encoder.

        Args:
            x_dict: Dict of node_type -> feature tensor [N, input_dim].
            edge_index_dict: Dict of (src, rel, dst) -> edge_index [2, E].

        Returns:
            Dict of node_type -> embedding tensor [N, output_dim].
        """
        # Input projection
        h_dict = {
            nt: self.input_projections[nt](x)
            for nt, x in x_dict.items()
            if nt in self.input_projections
        }

        # HGT layers
        for i, conv in enumerate(self.convs):
            h_dict_new = conv(h_dict, edge_index_dict)
            # Residual + LayerNorm
            h_dict = {
                nt: self.norms[i][nt](self.dropout(h_dict_new.get(nt, h)) + h)
                for nt, h in h_dict.items()
            }

        # Output projection
        out_dict = {
            nt: self.output_projection[nt](h)
            for nt, h in h_dict.items()
            if nt in self.output_projection
        }

        return out_dict


class MaskedNodePrediction(nn.Module):
    """Masked node feature reconstruction head (GraphMAE-style).

    Masks node features and reconstructs them from graph context.
    """

    def __init__(self, node_types: list[str], hidden_dim: int = 128):
        super().__init__()
        self.decoders = nn.ModuleDict({
            nt: nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                Linear(hidden_dim, hidden_dim),
            )
            for nt in node_types
        })

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        original_features: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute reconstruction loss on masked nodes.

        Args:
            embeddings: Encoder output per node type.
            original_features: Original (unmasked) features per node type.
            mask_dict: Boolean mask per node type (True = masked).

        Returns:
            Scalar reconstruction loss.
        """
        total_loss = 0.0
        count = 0

        for nt in embeddings:
            if nt not in mask_dict or mask_dict[nt].sum() == 0:
                continue

            mask = mask_dict[nt]
            pred = self.decoders[nt](embeddings[nt][mask])
            target = original_features[nt][mask]

            # Scaled cosine error (from GraphMAE)
            cos_sim = nn.functional.cosine_similarity(pred, target, dim=-1)
            loss = (1 - cos_sim).mean()
            total_loss += loss
            count += 1

        return total_loss / max(count, 1)


class LinkPrediction(nn.Module):
    """Link prediction head for edge existence prediction."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge existence probability.

        Args:
            src_emb: Source node embeddings [E, D].
            dst_emb: Destination node embeddings [E, D].

        Returns:
            Edge existence logits [E, 1].
        """
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.predictor(combined).squeeze(-1)


class TemporalOrdering(nn.Module):
    """Temporal ordering head — predict which of two nodes came first."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            Linear(hidden_dim, 1),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
    ) -> torch.Tensor:
        """Predict P(a appeared before b).

        Args:
            emb_a: Embeddings of first objects [B, D].
            emb_b: Embeddings of second objects [B, D].

        Returns:
            Logits [B] (positive = a before b).
        """
        combined = torch.cat([emb_a, emb_b], dim=-1)
        return self.predictor(combined).squeeze(-1)


class STIXBert(nn.Module):
    """Full STIXBert model with encoder + all pre-training heads."""

    def __init__(
        self,
        node_types: list[str],
        edge_types: list[tuple[str, str, str]],
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = STIXBertEncoder(
            node_types=node_types,
            edge_types=edge_types,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.masked_node_head = MaskedNodePrediction(node_types, hidden_dim)
        self.link_pred_head = LinkPrediction(hidden_dim)
        self.temporal_head = TemporalOrdering(hidden_dim)

    def forward(self, x_dict, edge_index_dict):
        """Encode graph and return node embeddings."""
        return self.encoder(x_dict, edge_index_dict)

    def get_embeddings(self, x_dict, edge_index_dict):
        """Get node embeddings (inference mode)."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x_dict, edge_index_dict)
