"""UMAP visualization for SimCLR-style models (encoder or projection embeddings)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch.utils.data import DataLoader, Dataset
from umap import UMAP


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _collate_batch_to_images_labels(batch):
    """Supports (x, y), ((x0, x1), y), or nested tuples from custom datasets."""
    inputs, labels = batch[0], batch[1]
    if isinstance(inputs, torch.Tensor):
        return inputs, labels
    if isinstance(inputs, (list, tuple)) and len(inputs) >= 1:
        first = inputs[0]
        if isinstance(first, torch.Tensor):
            return first, labels
    raise TypeError(
        "Expected batch[0] to be a Tensor or a tuple/list whose first element is a Tensor."
    )


def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    representation: Literal["encoder", "projection"],
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    core = _unwrap(model)
    all_emb: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    seen = 0

    with torch.no_grad():
        for batch in loader:
            images, labels = _collate_batch_to_images_labels(batch)
            images = images.to(device, non_blocking=True)

            if representation == "encoder":
                if not hasattr(core, "encoder"):
                    raise AttributeError(
                        "representation='encoder' requires model.encoder (SimCLR-style)."
                    )
                h = core.encoder(images)
                emb = torch.flatten(h, start_dim=1)
            else:
                emb = model(images)
                if emb.dim() > 2:
                    emb = torch.flatten(emb, start_dim=1)

            all_emb.append(emb.cpu())
            all_y.append(labels.cpu() if torch.is_tensor(labels) else torch.as_tensor(labels))

            seen += emb.shape[0]
            if max_samples is not None and seen >= max_samples:
                break

    embeddings = torch.cat(all_emb, dim=0).numpy()
    y = torch.cat(all_y, dim=0).numpy()

    if max_samples is not None and embeddings.shape[0] > max_samples:
        embeddings = embeddings[:max_samples]
        y = y[:max_samples]

    return embeddings, y


class SimCLRUmapVisualizer:
    """Fit UMAP on embeddings from a SimCLR-style model and plot class structure."""

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_components: int = 2,
        metric: str = "cosine",
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        self._umap = UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=verbose,
        )
        self._last_embedding_2d: np.ndarray | None = None
        self._last_labels_: np.ndarray | None = None

    @property
    def reducer(self) -> UMAP:
        return self._umap

    def visualize(
        self,
        model: nn.Module,
        dataset: Dataset,
        *,
        device: torch.device | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        max_samples: int | None = 5000,
        representation: Literal["encoder", "projection"] = "encoder",
        title: str | None = None,
        point_size: float = 8.0,
        alpha: float = 0.65,
        cmap: str = "tab10",
        figsize: tuple[float, float] = (8.0, 6.0),
        save_path: str | None = None,
    ) -> Figure:
        """
        Encode ``dataset`` with ``model``, fit UMAP on embeddings, return a scatter Figure.

        ``representation``:
          - ``encoder``: backbone features (typical for class separation / linear probe space).
          - ``projection``: full SimCLR forward (NT-Xent space).
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        model = model.to(device)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        embeddings, labels = _extract_embeddings(
            model, loader, device, representation=representation, max_samples=max_samples
        )

        xy = self._umap.fit_transform(embeddings)
        self._last_embedding_2d = xy
        self._last_labels_ = labels

        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels,
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(title or f"UMAP ({representation})")
        fig.colorbar(scatter, ax=ax, label="label")
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


def compare_models_umap(
    models: dict[str, nn.Module],
    dataset: Dataset,
    visualizer: SimCLRUmapVisualizer | None = None,
    *,
    device: torch.device | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    max_samples: int | None = 5000,
    representation: Literal["encoder", "projection"] = "encoder",
    figsize_per_ax: tuple[float, float] = (5.0, 4.0),
    save_path: str | None = None,
) -> Figure:
    """
    One UMAP panel per named model (same dataset, same UMAP hyperparameters, separate fits).

    Use this to eyeball how differently trained SimCLR checkpoints organize the same data.
    """
    if not models:
        raise ValueError("models must be a non-empty dict name -> nn.Module")

    viz = visualizer or SimCLRUmapVisualizer()
    names = list(models.keys())
    n = len(names)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(figsize_per_ax[0] * n, figsize_per_ax[1]),
        squeeze=False,
    )
    row = axes[0]

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    for ax, name in zip(row, names, strict=True):
        m = models[name].to(device)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        embeddings, labels = _extract_embeddings(
            m, loader, device, representation=representation, max_samples=max_samples
        )
        reducer = UMAP(
            n_neighbors=viz.reducer.n_neighbors,
            min_dist=viz.reducer.min_dist,
            n_components=viz.reducer.n_components,
            metric=viz.reducer.metric,
            random_state=viz.reducer.random_state,
            verbose=viz.reducer.verbose,
        )
        xy = reducer.fit_transform(embeddings)
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels,
            cmap="tab10",
            s=6.0,
            alpha=0.65,
            linewidths=0,
        )
        ax.set_title(name)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=ax, label="label")

    fig.suptitle(f"UMAP comparison ({representation})")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def visualize_simclr_embeddings(
    model: nn.Module,
    dataset: Dataset,
    **kwargs,
) -> Figure:
    """Convenience: ``SimCLRUmapVisualizer().visualize(model, dataset, **kwargs)``."""
    return SimCLRUmapVisualizer().visualize(model, dataset, **kwargs)
