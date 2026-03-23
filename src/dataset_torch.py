import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class PairedDataset(Dataset):
    def __init__(self, dir_a, dir_b, img_size=256):
        self.files_a = sorted(Path(dir_a).glob("*.jpg"))
        self.dir_b = Path(dir_b)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),  # maps [0,1] → [-1,1]
        ])

    def __len__(self):
        return len(self.files_a)

    def __getitem__(self, idx):
        path_a = self.files_a[idx]
        path_b = self.dir_b / path_a.name.replace("_A.jpg", "_B.jpg")
        return (
            self.transform(Image.open(path_a).convert("RGB")),
            self.transform(Image.open(path_b).convert("RGB")),
        )


def get_dataloader(dir_a, dir_b, batch_size=2):
    ds = PairedDataset(dir_a, dir_b)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)