"""Callbacks que generan las gráficas de WandB: reconstrucciones, t-SNE del latente y errores/histogramas de prueba."""
import io
from pathlib import Path

import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torchvision.utils import make_grid


def fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def _log_image(trainer, key, img, caption):
    logger = trainer.logger
    if logger is not None and hasattr(logger, "log_image"):
        logger.log_image(key=key, images=[img], caption=[caption], step=trainer.global_step)


def _first_n(dataset, n):
    items = [dataset[i] for i in range(min(n, len(dataset)))]
    return torch.stack([it["image"] for it in items])


@torch.no_grad()
def _encode_all(pl_module, loader):
    zs, labels, cats = [], [], []
    for batch in loader:
        out = pl_module(batch["image"].to(pl_module.device))
        zs.append(out["z"].cpu())
        labels += [int(l) for l in batch["label"]]
        cats += list(batch["category"])
    return torch.cat(zs).numpy(), np.array(labels), np.array(cats)


class ReconstructionCallback(L.Callback):
    def __init__(self, every_n_epochs=1, n_images=16):
        self.every_n, self.n = every_n_epochs, n_images

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n:
            return
        x = _first_n(trainer.datamodule.val_set, self.n).to(pl_module.device)
        was_training = pl_module.training
        pl_module.eval()
        x_hat = pl_module(x)["x_hat"]
        pl_module.train(was_training)
        pairs = torch.stack([x, x_hat], dim=1).flatten(0, 1)  # o1, r1, o2, r2...
        grid = make_grid(pairs.cpu(), nrow=8, padding=2, pad_value=1.0)
        img = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        _log_image(trainer, "val/reconstrucciones", img,
                   f"Época {trainer.current_epoch + 1} — pares original | reconstrucción")


class LatentTSNECallback(L.Callback):
    def __init__(self, every_n_epochs=10):
        self.every_n = every_n_epochs

    def _should_run(self, trainer):
        ep = trainer.current_epoch + 1
        return ep % self.every_n == 0 or ep == trainer.max_epochs

    @staticmethod
    def _tsne(z):
        perplexity = max(2, min(30, len(z) - 1))
        return TSNE(n_components=2, perplexity=perplexity, init="pca",
                    random_state=0).fit_transform(z)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not self._should_run(trainer):
            return
        dm, ep = trainer.datamodule, trainer.current_epoch + 1
        pl_module.eval()
        z_val, _, cat_val = _encode_all(pl_module, dm.val_dataloader())
        z_test, lab_test, cat_test = _encode_all(pl_module, dm.test_dataloader())
        pl_module.train()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        emb = self._tsne(z_val)
        for c in np.unique(cat_val):
            m = cat_val == c
            axes[0].scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7, label=c)
        axes[0].set_title(f"t-SNE latente — Validación (por categoría) — Época {ep}")
        axes[0].legend()

        emb = self._tsne(z_test)
        for lab, color, name in [(0, "green", "Good"), (1, "red", "Defective")]:
            m = lab_test == lab
            axes[1].scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.6, c=color, label=name)
        axes[1].set_title(f"t-SNE latente — Prueba (bueno vs defecto) — Época {ep}")
        axes[1].legend()
        for ax in axes:
            ax.set_xlabel("TSNE 1"); ax.set_ylabel("TSNE 2"); ax.grid(alpha=0.3)
        _log_image(trainer, "latente/tsne", fig_to_pil(fig), f"Época {ep}")


class TestEvaluationCallback(L.Callback):
    def __init__(self, out_dir=".", n_images=16):
        self.out_dir, self.n = Path(out_dir), n_images

    @torch.no_grad()
    def _test_reconstructions(self, trainer, pl_module):
        ds = trainer.datamodule.test_set
        cats = sorted({s[2] for s in ds.samples})
        per_cat = max(1, self.n // (2 * len(cats)))
        idx = []
        for label in (0, 1):  # primero buenas, luego defectuosas
            for c in cats:
                cand = [i for i, s in enumerate(ds.samples) if s[2] == c and s[1] == label]
                step = max(1, len(cand) // per_cat)
                idx += cand[::step][:per_cat]
        x = torch.stack([ds[i]["image"] for i in idx]).to(pl_module.device)
        x_hat = pl_module(x)["x_hat"]
        err = (x - x_hat).abs().mean(1)

        n = len(idx)
        fig, axes = plt.subplots(3, n, figsize=(1.6 * n, 5.4))
        for j, i in enumerate(idx):
            _, label, cat, defect = ds.samples[i]
            axes[0, j].imshow(x[j].permute(1, 2, 0).cpu().numpy())
            axes[0, j].set_title(f"{cat}\n{defect}", fontsize=7,
                                 color="green" if label == 0 else "red")
            axes[1, j].imshow(x_hat[j].permute(1, 2, 0).cpu().numpy())
            axes[2, j].imshow(err[j].cpu().numpy(), cmap="hot", vmin=0, vmax=0.5)
        for r, name in enumerate(["Original", "Reconstrucción", "|Error|"]):
            axes[r, 0].set_ylabel(name, fontsize=9)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("Prueba: buenas (verde) vs con defecto (rojo)")
        _log_image(trainer, "test/reconstrucciones", fig_to_pil(fig),
                   "Fila 1 original, fila 2 reconstrucción, fila 3 mapa de error")

    def _histograms(self, trainer, df, metric="mse"):
        cats = sorted(df.category.unique())
        fig, axes = plt.subplots(2, (len(cats) + 1) // 2, figsize=(14, 9))
        for ax, c in zip(np.ravel(axes), cats):
            d = df[df.category == c]
            bins = np.histogram_bin_edges(d[metric], bins=30)
            ax.hist(d[d.defect == "good"][metric], bins=bins, alpha=0.6,
                    color="green", label="good", density=True)
            for defect in sorted(d[d.label == 1].defect.unique()):
                ax.hist(d[d.defect == defect][metric], bins=bins, alpha=0.45,
                        histtype="stepfilled", label=defect, density=True)
            ax.set_title(c); ax.set_xlabel(f"Error de reconstrucción ({metric})")
            ax.legend(fontsize=7)
        fig.suptitle("Error de reconstrucción: normal vs cada subclase de defecto")
        fig.tight_layout()
        _log_image(trainer, f"test/histogramas_{metric}", fig_to_pil(fig), metric)

    def on_test_epoch_end(self, trainer, pl_module):
        df = pd.DataFrame(pl_module.test_records)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_dir / "test_errors.csv", index=False)

        pl_module.eval()
        self._test_reconstructions(trainer, pl_module)
        self._histograms(trainer, df, "mse")
        self._histograms(trainer, df, "ssim_err")

        # AUROC: qué tan bien separa el error a las buenas de las defectuosas
        metrics = {}
        for metric in ("mse", "ssim_err"):
            for c in sorted(df.category.unique()):
                d = df[df.category == c]
                if d.label.nunique() == 2:
                    metrics[f"test/auroc_{metric}_{c}"] = roc_auc_score(d.label, d[metric])
            metrics[f"test/auroc_{metric}_mean"] = float(np.mean(
                [v for k, v in metrics.items() if k.startswith(f"test/auroc_{metric}_")
                 and not k.endswith("mean")]))
        pl_module.log_dict(metrics)

        if trainer.logger is not None and hasattr(trainer.logger, "log_table"):
            trainer.logger.log_table(key="test/errores_por_imagen", dataframe=df)
