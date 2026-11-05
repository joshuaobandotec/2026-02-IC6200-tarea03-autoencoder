"""LightningModule compartido entre VAE y U-Net: recibe el modelo armado y maneja train/val/test, la pérdida de Hydra y el KL del VAE."""
import lightning as L
import torch
import torch.nn.functional as F

from src.losses import kl_divergence, reconstruction_loss, ssim


class AutoencoderModule(L.LightningModule):
    def __init__(self, net, loss="l1", ssim_alpha=0.85, kl_weight=0.0, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net
        self.test_records = []  # errores por imagen del set de prueba

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, stage):
        x = batch["image"]
        out = self(x)
        x_hat = out["x_hat"]
        rec = reconstruction_loss(self.hparams.loss, x_hat, x, self.hparams.ssim_alpha)
        loss = rec
        bs = x.size(0)
        if "mu" in out:  # VAE
            kl = kl_divergence(out["mu"], out["logvar"])
            loss = rec + self.hparams.kl_weight * kl
            self.log(f"{stage}/kl", kl, batch_size=bs)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True,
                 on_step=(stage == "train"), batch_size=bs)
        self.log(f"{stage}/rec_loss", rec, on_epoch=True, batch_size=bs)
        if stage == "val":  # métricas comunes para comparar entre pérdidas
            self.log("val/mse", F.mse_loss(x_hat, x), batch_size=bs)
            self.log("val/l1", F.l1_loss(x_hat, x), batch_size=bs)
            self.log("val/ssim", ssim(x_hat, x), batch_size=bs)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    # ---------- Prueba: error de reconstrucción por imagen ----------
    def on_test_epoch_start(self):
        self.test_records = []

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        x_hat = self(x)["x_hat"]
        mse = ((x_hat - x) ** 2).flatten(1).mean(1)
        l1 = (x_hat - x).abs().flatten(1).mean(1)
        ssim_each = torch.stack([ssim(x_hat[i:i + 1], x[i:i + 1]) for i in range(x.size(0))])
        for i in range(x.size(0)):
            self.test_records.append({
                "category": batch["category"][i], "defect": batch["defect"][i],
                "label": int(batch["label"][i]), "mse": mse[i].item(),
                "l1": l1[i].item(), "ssim_err": 1.0 - ssim_each[i].item(),
            })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
