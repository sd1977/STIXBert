"""Fine-tuning heads for downstream tasks."""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import Linear

logger = logging.getLogger(__name__)


class ATTACKClassifier(nn.Module):
    """Fine-tuning head for ATT&CK technique classification.

    Takes pre-trained node embeddings and predicts ATT&CK technique labels.
    """

    def __init__(self, input_dim: int, num_techniques: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(input_dim, num_techniques),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class ReputationScorer(nn.Module):
    """Fine-tuning head for multi-feed reputation scoring.

    Predicts a unified reputation score [0, 1] from node embeddings.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(input_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.scorer(embeddings).squeeze(-1)


class CampaignClusterer:
    """Campaign clustering using pre-trained embeddings.

    Not a trainable module — uses embeddings directly for clustering.
    """

    def __init__(self, n_clusters: int = 10, method: str = "kmeans"):
        self.n_clusters = n_clusters
        self.method = method

    def fit_predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cluster embeddings and return cluster assignments.

        Args:
            embeddings: Node embeddings [N, D].

        Returns:
            Cluster labels [N].
        """
        from sklearn.cluster import KMeans, HDBSCAN

        X = embeddings.cpu().numpy()

        if self.method == "kmeans":
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
        elif self.method == "hdbscan":
            clusterer = HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return torch.tensor(labels)
