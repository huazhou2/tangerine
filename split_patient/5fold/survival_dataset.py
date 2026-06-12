"""
Dataset for 6-year survival prediction with TANGERINE.
Reads pre-computed y_seq / y_mask columns from the split CSVs.
Weighted sampler still based on cancer=1 flag for class balance.
"""
import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


MAX_FOLLOWUP = 6


class LungCancerSurvivalDataset(Dataset):

    def __init__(self, csv_file, images_dir, patch_size=(256, 256, 256),
                 augment=False, mode='train'):
        self.df         = pd.read_csv(csv_file)
        if self.df.columns[0].startswith('Unnamed'):
            self.df = self.df.drop(columns=[self.df.columns[0]])

        self.images_dir = Path(images_dir)
        self.patch_size = patch_size
        self.augment    = augment and (mode == 'train')
        self.mode       = mode

        for col in ['ct_id', 'AnonAcc', 'MRN']:
            if col in self.df.columns:
                self.id_col = col
                break
        else:
            raise ValueError("No ID column found")

        print(f"[{mode.upper()}] {len(self.df)} samples  "
              f"(cancer+={int(self.df['cancer'].sum())}  "
              f"cancer-={int((self.df['cancer']==0).sum())})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── Load volume ────────────────────────────────────────────────
        vol = sitk.GetArrayFromImage(
            sitk.ReadImage(str(self.images_dir / row['image_filename']))
        ).astype(np.float32)

        while vol.ndim > 3:
            vol = vol.squeeze()

        # ── Crop ───────────────────────────────────────────────────────
        vol = self._random_crop(vol) if self.mode == 'train' \
              else self._center_crop(vol)

        if self.augment:
            vol = self._augment(vol)

        vol = vol[np.newaxis]  # (1, D, H, W)

        # ── Survival labels ────────────────────────────────────────────
        y_seq  = np.array([row[f'y_seq_{t}']  for t in range(MAX_FOLLOWUP)],
                          dtype=np.float32)
        y_mask = np.array([row[f'y_mask_{t}'] for t in range(MAX_FOLLOWUP)],
                          dtype=np.float32)

        return {
            'volume':         torch.from_numpy(vol),
            'y_seq':          torch.from_numpy(y_seq),
            'y_mask':         torch.from_numpy(y_mask),
            'cancer':         torch.tensor(int(row['cancer']),        dtype=torch.long),
            'time_at_event':  torch.tensor(float(row['time_at_event']), dtype=torch.float32),
            'patient_id':     str(row[self.id_col]),
        }

    # ── Crop helpers ───────────────────────────────────────────────────
    def _pad_if_needed(self, vol):
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            vol = np.pad(vol,
                         ((0, max(0, pd-d)), (0, max(0, ph-h)), (0, max(0, pw-w))),
                         mode='constant', constant_values=vol.min())
        return vol

    def _random_crop(self, vol):
        vol = self._pad_if_needed(vol)
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        ds = np.random.randint(0, d - pd + 1)
        hs = np.random.randint(0, h - ph + 1)
        ws = np.random.randint(0, w - pw + 1)
        return vol[ds:ds+pd, hs:hs+ph, ws:ws+pw]

    def _center_crop(self, vol):
        vol = self._pad_if_needed(vol)
        d, h, w = vol.shape
        pd, ph, pw = self.patch_size
        return vol[(d-pd)//2:(d-pd)//2+pd,
                   (h-ph)//2:(h-ph)//2+ph,
                   (w-pw)//2:(w-pw)//2+pw]

    def _augment(self, vol):
        if np.random.rand() > 0.5: vol = np.flip(vol, 1).copy()
        if np.random.rand() > 0.5: vol = np.flip(vol, 2).copy()
        if np.random.rand() > 0.5: vol = vol + np.random.uniform(-0.1, 0.1)
        if np.random.rand() > 0.5: vol = vol * np.random.uniform(0.9, 1.1)
        return vol


def create_survival_dataloaders(dataset_dir, images_dir, batch_size=4,
                                num_workers=8, patch_size=(256,256,256),
                                augment_train=True):
    dpath = Path(dataset_dir)

    train_ds = LungCancerSurvivalDataset(dpath/'train.csv', images_dir,
                                         patch_size, augment_train, 'train')
    val_ds   = LungCancerSurvivalDataset(dpath/'val.csv',   images_dir,
                                         patch_size, False, 'val')
    test_ds  = LungCancerSurvivalDataset(dpath/'test.csv',  images_dir,
                                         patch_size, False, 'test')

    # Weighted sampler: oversample cancer=1
    labels        = train_ds.df['cancer'].values
    class_counts  = np.bincount(labels)
    sample_w      = (1.0 / class_counts)[labels]
    sampler       = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
    print(f"\nWeighted sampler: neg={class_counts[0]}  pos={class_counts[1]}  "
          f"ratio={class_counts[0]/class_counts[1]:.1f}:1")

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
