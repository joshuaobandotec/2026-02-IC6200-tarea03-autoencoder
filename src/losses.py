"""Funciones de pérdida de reconstrucción: L1, L2, SSIM y SSIM + L1 (Persona A)."""
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn


def ssim(x_hat, x):
    return ssim_fn(x_hat, x, data_range=1.0)


def reconstruction_loss(name, x_hat, x, alpha=0.85):
    if name == "l1":
        return F.l1_loss(x_hat, x)
    if name == "l2":
        return F.mse_loss(x_hat, x)
    if name == "ssim":
        return 1.0 - ssim(x_hat, x)
    if name == "ssim_l1":
        return alpha * (1.0 - ssim(x_hat, x)) + (1.0 - alpha) * F.l1_loss(x_hat, x)
    raise ValueError(f"Pérdida desconocida: {name}. Usa l1 | l2 | ssim | ssim_l1")


def kl_divergence(mu, logvar):
    """KL(q(z|x) || N(0, I)), promediada por dimensión latente y por batch."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
