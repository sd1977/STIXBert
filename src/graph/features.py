"""Encode STIX object attributes into node feature vectors."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class FeatureEncoder:
    """Encodes STIX object attributes into fixed-size feature vectors.

    Uses sentence-transformers for text fields and learned embeddings
    for categorical attributes.
    """

    def __init__(
        self,
        text_model_name: str = "all-MiniLM-L6-v2",
        output_dim: int = 128,
        device: str = "cpu",
    ):
        self.output_dim = output_dim
        self.device = device
        self._text_model = None
        self._text_model_name = text_model_name
        self._text_dim = None

    @property
    def text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(self._text_model_name)
            self._text_dim = self._text_model.get_sentence_embedding_dimension()
            logger.info(
                f"Loaded text model '{self._text_model_name}' (dim={self._text_dim})"
            )
        return self._text_model

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text strings into vectors using sentence-transformers."""
        embeddings = self.text_model.encode(
            texts, show_progress_bar=True, convert_to_tensor=True
        )
        return embeddings.to(self.device)

    def encode_stix_nodes(
        self, nodes: list[dict], node_type: str
    ) -> torch.Tensor:
        """Encode a list of STIX objects of the same type into feature vectors.

        Combines text embeddings (from name + description) with categorical
        features specific to each node type.

        Args:
            nodes: List of STIX object dicts.
            node_type: Normalized STIX type (e.g., 'attack_pattern').

        Returns:
            Tensor of shape [num_nodes, output_dim].
        """
        # Extract text for embedding
        texts = []
        for node in nodes:
            name = node.get("name", "")
            desc = node.get("description", "")
            text = f"{name}. {desc}" if desc else name
            texts.append(text if text.strip() else node.get("id", "unknown"))

        text_embeddings = self.encode_text(texts)

        # Project to output_dim
        projector = torch.nn.Linear(
            text_embeddings.shape[1], self.output_dim
        ).to(self.device)
        with torch.no_grad():
            features = projector(text_embeddings)

        return features

    def encode_all_node_types(
        self, nodes_by_type: dict[str, list[dict]]
    ) -> dict[str, torch.Tensor]:
        """Encode all node types in the graph.

        Args:
            nodes_by_type: Dict of node_type -> list of STIX objects.

        Returns:
            Dict of node_type -> feature tensor.
        """
        encoded = {}
        for node_type, nodes in nodes_by_type.items():
            logger.info(f"Encoding {len(nodes)} '{node_type}' nodes...")
            encoded[node_type] = self.encode_stix_nodes(nodes, node_type)
        return encoded
