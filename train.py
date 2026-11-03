"""Punto de entrada. Ejemplos:

    # un experimento
    python train.py model=vae loss=l1

    # los 4 experimentos de un modelo (Hydra multirun)
    python train.py -m model=vae loss=l1,l2,ssim,ssim_l1
    python train.py -m model=u-net loss=l1,l2,ssim,ssim_l1

    # cambiar hiperparámetros
    python train.py model=vae model.latent_dim=64 trainer.max_epochs=50 data.batch_size=64
"""
import hydra
import lightning as L
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src.callbacks import LatentTSNECallback, ReconstructionCallback, TestEvaluationCallback
from src.data import MVTecDataModule
from src.module import AutoencoderModule


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    L.seed_everything(cfg.seed, workers=True)
    out_dir = HydraConfig.get().runtime.output_dir

    # to_container: convierte las listas de Hydra a listas normales de Python
    # (si no, PyTorch >= 2.6 no puede volver a cargar el checkpoint)
    dm = MVTecDataModule(**OmegaConf.to_container(cfg.data, resolve=True))
    module = AutoencoderModule(
        net=instantiate(cfg.model.net), loss=cfg.loss, ssim_alpha=cfg.ssim_alpha,
        kl_weight=cfg.model.kl_weight, lr=cfg.lr,
    )

    logger = WandbLogger(
        project=cfg.logger.project, entity=cfg.logger.entity, group=cfg.logger.group,
        name=cfg.logger.name, mode=cfg.logger.mode, save_dir=out_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = [
        ModelCheckpoint(dirpath=f"{out_dir}/checkpoints", monitor="val/loss",
                        mode="min", save_top_k=1),
        ReconstructionCallback(cfg.callbacks.recon_every_n_epochs, cfg.callbacks.n_images),
        LatentTSNECallback(cfg.callbacks.tsne_every_n_epochs),
        TestEvaluationCallback(out_dir=out_dir, n_images=cfg.callbacks.n_images),
    ]

    trainer = L.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(module, datamodule=dm)
    trainer.test(module, datamodule=dm, ckpt_path="best")
    wandb.finish()  # necesario para que cada run del multirun quede separado


if __name__ == "__main__":
    main()
