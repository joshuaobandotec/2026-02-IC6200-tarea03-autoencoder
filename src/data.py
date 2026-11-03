"""DataModule de MVTec AD (Persona A).

Estructura esperada en disco (la del dataset oficial):
    root/<categoria>/train/good/*.png
    root/<categoria>/test/<good|tipo_defecto>/*.png

- train / val: solo imágenes buenas (el autoencoder aprende "lo normal").
- test: imágenes buenas y defectuosas, con su categoría y subclase de defecto.

Cada muestra es un diccionario:
    {"image": Tensor[3,H,W] en [0,1], "label": 0 bueno / 1 defecto,
     "category": str, "defect": str}

Las imágenes se cargan y redimensionan UNA sola vez a memoria (son ~1500
imágenes de 128x128), así cada época es rápida.
"""
from pathlib import Path

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


class MVTecDataset(Dataset):
    def __init__(self, root, categories, split, image_size=128):
        self.root = Path(root)
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
        ])
        self.samples = []  # (path, label, category, defect)
        for cat in categories:
            split_dir = self.root / cat / split
            if not split_dir.exists():
                raise FileNotFoundError(f"No existe {split_dir}. Revisa data.root")
            for defect_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
                defect = defect_dir.name
                label = 0 if defect == "good" else 1
                for p in _list_images(defect_dir):
                    self.samples.append((p, label, cat, defect))

        # Cache en memoria. convert("RGB") es necesario: screw viene en escala de grises.
        self.images = torch.stack(
            [tf(Image.open(p).convert("RGB")) for p, *_ in self.samples]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, label, cat, defect = self.samples[idx]
        return {"image": self.images[idx], "label": label,
                "category": cat, "defect": defect}


class MVTecDataModule(L.LightningDataModule):
    def __init__(self, root, categories, image_size=128, batch_size=32,
                 val_split=0.15, num_workers=2, seed=42):
        super().__init__()
        self.save_hyperparameters()
        self.train_set = self.val_set = self.test_set = None

    def setup(self, stage=None):
        h = self.hparams
        if self.train_set is None:
            full = MVTecDataset(h.root, h.categories, "train", h.image_size)
            n_val = max(1, int(len(full) * h.val_split))
            gen = torch.Generator().manual_seed(h.seed)
            self.train_set, self.val_set = random_split(
                full, [len(full) - n_val, n_val], generator=gen)
        if self.test_set is None:
            self.test_set = MVTecDataset(h.root, h.categories, "test", h.image_size)

    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._loader(self.train_set, True)

    def val_dataloader(self):
        return self._loader(self.val_set, False)

    def test_dataloader(self):
        return self._loader(self.test_set, False)
