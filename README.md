# Tarea 03 — Autoencoders (VAE y U-Net) en MVTec AD

PyTorch Lightning + Hydra + WandB.

## Estructura

```
conf/
  config.yaml          # pérdida, lr, semilla, frecuencia de gráficas
  model/vae.yaml       # VAE (latent_dim, kl_weight)
  model/u-net.yaml     # U-Net con skip connections (latent_dim, use_skips)
  trainer/default.yaml # épocas, GPU, etc. (iguales para todos los runs)
  logger/wandb.yaml    # proyecto / team de WandB
  data/mvtec.yaml      # ruta del dataset, categorías, 128x128, batch
src/
  data.py              # DataModule de MVTec (Persona A)
  losses.py            # L1, L2, SSIM, SSIM+L1, KL (Persona A)
  models/vae.py        # VAE (Persona A)
  models/unet.py       # U-Net autoencoder (Persona B)
  module.py            # LightningModule común (entrenamiento / métricas)
  callbacks.py         # gráficas en WandB: reconstrucciones, t-SNE, histogramas (Persona B)
train.py               # punto de entrada con Hydra
```

## Cómo correrlo en Colab (Entorno de ejecución → GPU T4)

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>
!pip install -q -r requirements.txt
import wandb; wandb.login()
```

Dataset: descargar de MVTec AD solo **cable, capsule, screw y transistor** y
descomprimir en `/content/drive/MyDrive/mvtec_ad/` (o pasar `data.root=...`).

```bash
# prueba rápida (2 épocas) antes de lanzar todo
!python train.py model=vae loss=l1 trainer.max_epochs=2

# los 4 experimentos de cada modelo
!python train.py -m model=vae   loss=l1,l2,ssim,ssim_l1
!python train.py -m model=u-net loss=l1,l2,ssim,ssim_l1

# con el team de WandB
!python train.py -m model=vae loss=l1,l2,ssim,ssim_l1 logger.entity=<team>
```

## Qué se registra en WandB

| Clave | Qué es |
|---|---|
| `train/loss`, `val/loss` | pérdida de entrenamiento y validación |
| `val/mse`, `val/l1`, `val/ssim` | métricas comunes para comparar pérdidas entre sí |
| `val/reconstrucciones` | 16 imágenes de validación: original \| reconstrucción |
| `latente/tsne` | t-SNE del latente (validación por categoría, prueba bueno vs defecto) cada N épocas |
| `test/reconstrucciones` | 8 buenas + 8 con defecto: original, reconstrucción, mapa de error |
| `test/histogramas_mse`, `test/histogramas_ssim_err` | error de reconstrucción normal vs cada subclase de defecto |
| `test/auroc_*` | qué tan bien separa el error a buenas de defectuosas, por categoría |
| `test/errores_por_imagen` | tabla con el error de cada imagen (también en `test_errors.csv`) |

Ojo: `val/loss` **no** es comparable entre pérdidas distintas (L1 y 1-SSIM
tienen escalas diferentes). Para comparar usar `val/mse`, `val/ssim` y AUROC.
