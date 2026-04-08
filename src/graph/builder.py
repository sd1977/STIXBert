"""Build PyG HeteroData graph from STIX 2.1 bundles."""

import logging
from collections import defaultdict
from typing import Optional

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# STIX relationship types we care about for graph construction
EDGE_TYPE_MAP = {
    "uses": "uses",
    "indicates": "indicates",
    "targets": "targets",
    "attributed-to": "attributed_to",
    "communicates-with": "communicates_with",
    "exploits": "exploits",
    "mitigates": "mitigates",
    "derived-from": "derived_from",
    "related-to": "related_to",
    "consists-of": "consists_of",
    "controls": "controls",
    "hosts": "hosts",
    "located-at": "located_at",
    "originates-from": "originates_from",
    "delivers": "delivers",
    "drops": "drops",
    "variant-of": "variant_of",
}

# STIX SDO types to include as graph nodes
SDO_NODE_TYPES = {
    "indicator",
    "malware",
    "attack-pattern",
    "threat-actor",
    "campaign",
    "intrusion-set",
    "infrastructure",
    "vulnerability",
    "tool",
    "identity",
    "location",
    "course-of-action",
}

# STIX SCO types to optionally include
SCO_NODE_TYPES = {
    "ipv4-addr",
    "ipv6-addr",
    "domain-name",
    "url",
    "file",
    "email-addr",
    "autonomous-system",
}


def normalize_type(stix_type: str) -> str:
    """Normalize STIX type to valid PyG node type (replace hyphens)."""
    return stix_type.replace("-", "_")


class STIXGraphBuilder:
    """Builds a PyG HeteroData graph from multiple STIX bundles."""

    def __init__(self, include_scos: bool = False):
        self.include_scos = include_scos
        self.allowed_types = SDO_NODE_TYPES.copy()
        if include_scos:
            self.allowed_types |= SCO_NODE_TYPES

        # Maps: stix_id -> (node_type, local_index)
        self.node_registry: dict[str, tuple[str, int]] = {}
        # Maps: node_type -> list of stix objects
        self.nodes_by_type: dict[str, list[dict]] = defaultdict(list)
        # Maps: (src_type, edge_type, dst_type) -> list of (src_idx, dst_idx)
        self.edges: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        # All relationship objects
        self.relationships: list[dict] = []
        # All sighting objects
        self.sightings: list[dict] = []

    def add_bundle(self, bundle: dict, source_name: str = "unknown") -> None:
        """Add all objects from a STIX bundle to the graph.

        Args:
            bundle: STIX 2.1 bundle dict.
            source_name: Name of the feed/source for provenance tracking.
        """
        objects = bundle.get("objects", [])
        logger.info(f"Processing bundle from '{source_name}': {len(objects)} objects")

        for obj in objects:
            obj_type = obj.get("type", "")
            obj_id = obj.get("id", "")

            if obj_type == "relationship":
                self.relationships.append(obj)
            elif obj_type == "sighting":
                self.sightings.append(obj)
            elif obj_type in self.allowed_types:
                self._register_node(obj, source_name)

        # Process relationships after all nodes are registered
        self._process_relationships()
        self._process_sightings()

    def _register_node(self, obj: dict, source_name: str) -> None:
        node_type = normalize_type(obj["type"])
        stix_id = obj["id"]

        if stix_id in self.node_registry:
            return  # Dedup by STIX ID

        local_idx = len(self.nodes_by_type[node_type])
        self.node_registry[stix_id] = (node_type, local_idx)
        obj["_source"] = source_name
        self.nodes_by_type[node_type].append(obj)

    def _process_relationships(self) -> None:
        for rel in self.relationships:
            src_id = rel.get("source_ref", "")
            dst_id = rel.get("target_ref", "")
            rel_type = rel.get("relationship_type", "")

            if src_id not in self.node_registry or dst_id not in self.node_registry:
                continue

            edge_type = EDGE_TYPE_MAP.get(rel_type, rel_type.replace("-", "_"))
            src_node_type, src_idx = self.node_registry[src_id]
            dst_node_type, dst_idx = self.node_registry[dst_id]

            edge_key = (src_node_type, edge_type, dst_node_type)
            self.edges[edge_key].append((src_idx, dst_idx))

        self.relationships.clear()

    def _process_sightings(self) -> None:
        for sighting in self.sightings:
            sighting_of = sighting.get("sighting_of_ref", "")
            observed_data_refs = sighting.get("observed_data_refs", [])
            where_sighted = sighting.get("where_sighted_refs", [])

            if sighting_of not in self.node_registry:
                continue

            src_type, src_idx = self.node_registry[sighting_of]

            for ref in where_sighted:
                if ref in self.node_registry:
                    dst_type, dst_idx = self.node_registry[ref]
                    edge_key = (src_type, "sighted_by", dst_type)
                    self.edges[edge_key].append((src_idx, dst_idx))

        self.sightings.clear()

    def build(self, feature_dim: int = 128) -> HeteroData:
        """Build the final PyG HeteroData object.

        Args:
            feature_dim: Dimension of initial node feature vectors.
                Features are initialized randomly; replace with encoded
                features from FeatureEncoder before training.

        Returns:
            PyG HeteroData graph.
        """
        data = HeteroData()

        # Add nodes with placeholder features
        for node_type, nodes in self.nodes_by_type.items():
            n = len(nodes)
            data[node_type].x = torch.randn(n, feature_dim)
            data[node_type].num_nodes = n
            logger.info(f"  Node type '{node_type}': {n} nodes")

        # Add edges
        for (src_type, edge_type, dst_type), edge_list in self.edges.items():
            if not edge_list:
                continue
            src_indices, dst_indices = zip(*edge_list)
            edge_index = torch.tensor(
                [list(src_indices), list(dst_indices)], dtype=torch.long
            )
            data[src_type, edge_type, dst_type].edge_index = edge_index
            logger.info(
                f"  Edge type ({src_type}, {edge_type}, {dst_type}): {len(edge_list)} edges"
            )

        return data

    def get_stats(self) -> dict:
        """Return graph statistics."""
        total_nodes = sum(len(v) for v in self.nodes_by_type.values())
        total_edges = sum(len(v) for v in self.edges.values())
        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "node_types": {k: len(v) for k, v in self.nodes_by_type.items()},
            "edge_types": {
                f"({s},{e},{d})": len(v) for (s, e, d), v in self.edges.items()
            },
        }
