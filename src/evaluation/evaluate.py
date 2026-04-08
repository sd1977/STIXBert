"""Evaluation and visualization tools for STIXBert demos."""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def compute_clustering_metrics(
    predicted_labels: torch.Tensor,
    ground_truth_labels: torch.Tensor,
) -> dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        predicted_labels: Predicted cluster assignments.
        ground_truth_labels: Ground truth labels.

    Returns:
        Dict with ARI, NMI, and homogeneity scores.
    """
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        homogeneity_score,
    )

    pred = predicted_labels.cpu().numpy()
    truth = ground_truth_labels.cpu().numpy()

    return {
        "ari": adjusted_rand_score(truth, pred),
        "nmi": normalized_mutual_info_score(truth, pred),
        "homogeneity": homogeneity_score(truth, pred),
    }


def plot_embeddings_umap(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    label_names: Optional[dict] = None,
    title: str = "STIXBert Embeddings",
    save_path: Optional[str] = None,
):
    """Plot 2D UMAP projection of node embeddings.

    Args:
        embeddings: Node embeddings [N, D].
        labels: Optional cluster/category labels [N].
        label_names: Optional mapping from label int to display name.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt
    from umap import UMAP

    X = embeddings.cpu().numpy()
    reducer = UMAP(n_components=2, random_state=42)
    X_2d = reducer.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    if labels is not None:
        y = labels.cpu().numpy()
        unique_labels = sorted(set(y))
        for label in unique_labels:
            mask = y == label
            name = label_names.get(label, f"Cluster {label}") if label_names else f"Cluster {label}"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, alpha=0.6, s=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=20)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")

    plt.show()
    return fig


def plot_cross_feed_overlap(
    embeddings_by_feed: dict[str, torch.Tensor],
    threshold: float = 0.9,
    save_path: Optional[str] = None,
):
    """Plot cross-feed overlap heatmap based on embedding similarity.

    Args:
        embeddings_by_feed: Dict of feed_name -> embeddings [N, D].
        threshold: Cosine similarity threshold for "overlap".
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    feed_names = list(embeddings_by_feed.keys())
    n_feeds = len(feed_names)
    overlap_matrix = np.zeros((n_feeds, n_feeds))

    for i, name_i in enumerate(feed_names):
        for j, name_j in enumerate(feed_names):
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue

            emb_i = embeddings_by_feed[name_i]
            emb_j = embeddings_by_feed[name_j]

            # Normalize
            emb_i_norm = emb_i / emb_i.norm(dim=1, keepdim=True)
            emb_j_norm = emb_j / emb_j.norm(dim=1, keepdim=True)

            # Compute max similarity for each indicator in feed i
            sim = torch.mm(emb_i_norm, emb_j_norm.t())
            max_sim, _ = sim.max(dim=1)
            overlap = (max_sim > threshold).float().mean().item()
            overlap_matrix[i, j] = overlap

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(n_feeds))
    ax.set_yticks(range(n_feeds))
    ax.set_xticklabels(feed_names, rotation=45, ha="right")
    ax.set_yticklabels(feed_names)

    for i in range(n_feeds):
        for j in range(n_feeds):
            ax.text(j, i, f"{overlap_matrix[i, j]:.0%}",
                    ha="center", va="center", fontsize=9)

    ax.set_title(f"Cross-Feed Overlap (cosine sim > {threshold})")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig, overlap_matrix


def plot_training_losses(
    history: list[dict],
    save_path: Optional[str] = None,
):
    """Plot training loss curves.

    Args:
        history: List of per-epoch loss dicts from PreTrainer.train().
        save_path: Optional path to save figure.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(history) + 1)
    keys = ["total", "masked_node", "link_pred", "temporal"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, key in zip(axes, keys):
        values = [h[key] for h in history]
        ax.plot(epochs, values)
        ax.set_title(f"{key} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    plt.suptitle("STIXBert Pre-training Losses")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    return fig


def label_efficiency_experiment(
    encoder,
    classifier_class,
    data,
    labels,
    node_type: str,
    fractions: list[float] = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
    pretrained: bool = True,
    num_trials: int = 3,
    device: str = "cuda",
) -> dict[float, float]:
    """Run label efficiency experiment.

    Trains classifier with varying fractions of labeled data and reports
    accuracy for each fraction. Compares pretrained vs random init.

    Args:
        encoder: STIXBert encoder (pretrained or randomly initialized).
        classifier_class: Classifier head class.
        data: PyG HeteroData.
        labels: Ground truth labels for node_type.
        node_type: Which node type to classify.
        fractions: List of label fractions to test.
        pretrained: Whether the encoder is pretrained.
        num_trials: Number of random trials per fraction.
        device: Device to train on.

    Returns:
        Dict mapping fraction -> mean accuracy.
    """
    import random
    from sklearn.model_selection import train_test_split

    results = {}
    n = labels.shape[0]

    # Get embeddings from encoder
    encoder.eval()
    with torch.no_grad():
        x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
        ei_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
        embeddings = encoder(x_dict, ei_dict)[node_type]

    num_classes = labels.max().item() + 1

    for frac in fractions:
        accs = []
        for trial in range(num_trials):
            n_train = max(10, int(n * frac))
            indices = list(range(n))
            random.shuffle(indices)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]

            if not test_idx:
                test_idx = train_idx  # fallback for frac=1.0

            clf = classifier_class(embeddings.shape[1], num_classes).to(device)
            opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

            # Train
            clf.train()
            for _ in range(100):
                logits = clf(embeddings[train_idx])
                loss = nn.CrossEntropyLoss()(logits, labels[train_idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Eval
            clf.eval()
            with torch.no_grad():
                pred = clf(embeddings[test_idx]).argmax(dim=1)
                acc = (pred == labels[test_idx].to(device)).float().mean().item()
            accs.append(acc)

        results[frac] = sum(accs) / len(accs)
        tag = "pretrained" if pretrained else "scratch"
        logger.info(f"[{tag}] Fraction {frac:.0%}: accuracy = {results[frac]:.3f}")

    return results
