#!/usr/bin/env python3
"""
train_ensemble_models.py

Train multiple base models with Stratified K-Fold and save OOF probabilities + averaged test probabilities.

Usage examples (from a Kaggle notebook cell prefix with !):
!python train_ensemble_models.py --models convnext_base efficientnet_b3 --n_splits 5 --epochs 8 --batch_size 16 --num_workers 4
!python train_ensemble_models.py --models convnext_base --n_splits 2 --epochs 1 --batch_size 8 --num_workers 0 --holdout_valid

Notes:
- This script expects fix_transforms.py to exist in the same directory and to provide get_transforms(img_size).
- By default it concatenates TRAIN + VALID CSVs for K-Fold. Use --holdout_valid to keep VALID as a separate holdout.
- Outputs are saved to `ensemble_outputs/`:
  - oof_probs_{model}.npy
  - test_probs_{model}.npy
  - meta_{model}.csv
  - per-fold checkpoints: {model}_fold{f}_best.pth
"""
import os
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.model_selection import StratifiedKFold

from fix_transforms import get_transforms

# -------------------
# Paths - adjust if your dataset is in a different location
# -------------------
TRAIN_CSV = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/train/_classes.csv"
TRAIN_IMG_DIR = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/train/"
VALID_CSV = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/valid/_classes.csv"
VALID_IMG_DIR = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/valid/"
TEST_CSV = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/test/_classes.csv"
TEST_IMG_DIR = "/kaggle/input/hairfall/hair-loss.v2i.multiclass/test/"

# -------------------
NUM_CLASSES = 6
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "ensemble_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Reproducibility helper
# -------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # keep deterministic=False for speed; change if you require strict determinism
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)

# -------------------
# Robust dataset: always returns a torch.Tensor of shape (3, IMG_SIZE, IMG_SIZE),
# plain Python ints for labels and indices, and filename string.
# It also accepts per-sample image directory mapping so train/valid can live in different folders.
# -------------------
class IndexedDataset(Dataset):
    def __init__(self, filenames, labels, img_dirs, transform=None, img_size=IMG_SIZE):
        """
        filenames: iterable of filename strings (as in CSV)
        labels: iterable of ints (or zeros for test)
        img_dirs: iterable of directory paths (same length) to join with filename
        transform: albumentations transform (should end with ToTensorV2)
        img_size: enforced output size (H==W==img_size)
        """
        assert len(filenames) == len(img_dirs)
        self.filenames = np.array(filenames)
        self.labels = np.array(labels)
        self.img_dirs = np.array(img_dirs)
        self.transform = transform
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        idx = int(idx)
        fname = str(self.filenames[idx])
        img_dir = str(self.img_dirs[idx])
        label = int(self.labels[idx])  # plain python int

        path = os.path.join(img_dir, fname)
        # Load image safely
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Try filename as absolute / relative path in case CSV contains subpaths
            try:
                img = Image.open(fname).convert("RGB")
            except Exception as e:
                # If loading fails, log and use a blank image fallback
                print(f"[IndexedDataset] Warning: could not open image {path} or {fname}: {e}")
                img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        # If very small, resize to avoid unexpected shapes from transforms
        try:
            if img.size[0] < self.img_size or img.size[1] < self.img_size:
                img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        except Exception:
            img = Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

        img_np = np.array(img)
        out = None
        if self.transform is not None:
            try:
                out = self.transform(image=img_np)["image"]
            except Exception as e:
                print(f"[IndexedDataset] Warning: transform failed for {path}: {e}")
                out = None

        # Convert transform output to torch tensor of shape (3, H, W) and dtype float32
        import torch as _torch
        if out is None:
            out_tensor = _torch.zeros((3, self.img_size, self.img_size), dtype=_torch.float32)
        else:
            if isinstance(out, _torch.Tensor):
                out_tensor = out
            else:
                # numpy array -> CHW torch tensor
                try:
                    arr = np.asarray(out)
                    # If HWC, reshape/rescale to expected
                    if arr.ndim == 3:
                        # Ensure spatial size
                        if arr.shape[0] != self.img_size and arr.shape[1] != self.img_size:
                            # assume HWC; if not, try to coerce via PIL
                            tmp = Image.fromarray(arr)
                            tmp = tmp.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
                            arr = np.array(tmp)
                        # HWC -> CHW
                        if arr.shape[2] == 3:
                            arr = arr.transpose(2, 0, 1)
                        elif arr.shape[0] == 3:
                            # already CHW
                            pass
                        else:
                            # unexpected channels, fallback
                            arr = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
                    else:
                        # unexpected dims
                        arr = np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)
                    out_tensor = _torch.from_numpy(arr).float()
                    # If values in 0..255 range, normalize to 0..1
                    if out_tensor.max() > 1.0:
                        out_tensor = out_tensor / 255.0
                except Exception as e:
                    print(f"[IndexedDataset] Warning: converting transform output failed for {path}: {e}")
                    out_tensor = _torch.zeros((3, self.img_size, self.img_size), dtype=_torch.float32)

        # Final safety: ensure shape is exactly (3, img_size, img_size)
        if out_tensor.dim() == 3:
            c, h, w = out_tensor.shape
            if (c, h, w) != (3, self.img_size, self.img_size):
                try:
                    arr = out_tensor.cpu().numpy().transpose(1, 2, 0)
                    tmp = Image.fromarray((arr * 255).astype("uint8")) if arr.max() <= 1.0 else Image.fromarray(arr.astype("uint8"))
                    tmp = tmp.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
                    arr2 = np.array(tmp).transpose(2, 0, 1)
                    out_tensor = _torch.from_numpy(arr2).float()
                    if out_tensor.max() > 1.0:
                        out_tensor = out_tensor / 255.0
                except Exception:
                    out_tensor = _torch.zeros((3, self.img_size, self.img_size), dtype=_torch.float32)
        else:
            out_tensor = _torch.zeros((3, self.img_size, self.img_size), dtype=_torch.float32)

        return out_tensor, label, int(idx), fname

# -------------------
# CSV helper
# -------------------
def read_csv_labels(csv_path):
    df = pd.read_csv(csv_path)
    if 'filename' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'filename'})
    labels = df.iloc[:, 1:].values.argmax(axis=1)
    return df['filename'].values, labels

# -------------------
# Training loop for a single fold
# -------------------
def train_fold(model, train_loader, val_loader, optimizer, scheduler, epochs, device, save_path):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * imgs.size(0)
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass
        avg_loss = running_loss / max(1, len(train_loader.dataset))

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    preds = model(imgs).argmax(1)
                correct += int((preds == labels).sum().item())
                total += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"  Epoch {epoch+1}/{epochs} - loss {avg_loss:.4f} - val_acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            try:
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                print(f"[train_fold] Warning: could not save model to {save_path}: {e}")

    print(f"  Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")
    return best_val_acc

# -------------------
# Run K-Fold for one model
# -------------------
def run_model_kfold(model_name, args):
    print(f"\n=== Running K-Fold for model: {model_name} ===")
    # Read CSVs
    train_fnames, train_labels = read_csv_labels(TRAIN_CSV)
    valid_fnames, valid_labels = read_csv_labels(VALID_CSV)
    test_fnames, test_labels = read_csv_labels(TEST_CSV)

    # Build combined lists and per-file img_dirs
    if args.holdout_valid:
        all_fnames = train_fnames
        all_labels = train_labels
        all_img_dirs = np.array([TRAIN_IMG_DIR] * len(all_fnames))
    else:
        all_fnames = np.concatenate([train_fnames, valid_fnames])
        all_labels = np.concatenate([train_labels, valid_labels])
        train_dirs = np.array([TRAIN_IMG_DIR] * len(train_fnames))
        valid_dirs = np.array([VALID_IMG_DIR] * len(valid_fnames))
        all_img_dirs = np.concatenate([train_dirs, valid_dirs])

    test_img_dirs = np.array([TEST_IMG_DIR] * len(test_fnames))

    # Transforms
    train_transform, valid_transform, test_transform = get_transforms(IMG_SIZE)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    oof_probs = np.zeros((len(all_fnames), NUM_CLASSES), dtype=np.float32)
    test_probs_accum = np.zeros((len(test_fnames), NUM_CLASSES), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_fnames, all_labels)):
        print(f"Fold {fold+1}/{args.n_splits}")

        tr_ds = IndexedDataset(all_fnames[tr_idx], all_labels[tr_idx], all_img_dirs[tr_idx], transform=train_transform, img_size=IMG_SIZE)
        vl_ds = IndexedDataset(all_fnames[val_idx], all_labels[val_idx], all_img_dirs[val_idx], transform=valid_transform, img_size=IMG_SIZE)
        test_ds = IndexedDataset(test_fnames, np.zeros(len(test_fnames)), test_img_dirs, transform=test_transform, img_size=IMG_SIZE)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers), pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=max(0, args.num_workers), pin_memory=True)

        # Build model
        print(f"  Building model {model_name} ...")
        model = timm.create_model(model_name, pretrained=True, num_classes=NUM_CLASSES)
        model.to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        steps_per_epoch = max(1, len(tr_loader))
        # Use OneCycleLR for stable training; stepping per batch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

        # Train
        fold_ckpt = os.path.join(OUTPUT_DIR, f"{model_name}_fold{fold+1}_best.pth")
        _ = train_fold(model, tr_loader, vl_loader, optimizer, scheduler, args.epochs, DEVICE, fold_ckpt)

        # Load best checkpoint if saved
        if os.path.exists(fold_ckpt):
            try:
                model.load_state_dict(torch.load(fold_ckpt, map_location=DEVICE))
                model.eval()
            except Exception as e:
                print(f"[run_model_kfold] Warning: failed to load checkpoint {fold_ckpt}: {e}")

        # Produce OOF probabilities for validation set and place into global oof_probs
        with torch.no_grad():
            for imgs, labels, local_idx, _ in vl_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                    probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
                local_idx = local_idx.numpy()
                for i, li in enumerate(local_idx):
                    global_index = int(val_idx[li])
                    oof_probs[global_index] = probs[i]

        # Produce test probs for this fold and accumulate (average over folds)
        fold_test_probs = []
        with torch.no_grad():
            for imgs, _, _, _ in test_loader:
                imgs = imgs.to(DEVICE)
                with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                    probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
                fold_test_probs.append(probs)
        if len(fold_test_probs):
            fold_test_probs = np.vstack(fold_test_probs)
            if fold_test_probs.shape[0] == test_probs_accum.shape[0]:
                test_probs_accum += fold_test_probs / args.n_splits
            else:
                m = min(fold_test_probs.shape[0], test_probs_accum.shape[0])
                test_probs_accum[:m] += fold_test_probs[:m] / args.n_splits

    # Save OOF and test probabilities and meta
    np.save(os.path.join(OUTPUT_DIR, f"oof_probs_{model_name}.npy"), oof_probs)
    np.save(os.path.join(OUTPUT_DIR, f"test_probs_{model_name}.npy"), test_probs_accum)
    meta_df = pd.DataFrame({"filename": all_fnames, "img_dir": all_img_dirs, "label": all_labels})
    meta_df.to_csv(os.path.join(OUTPUT_DIR, f"meta_{model_name}.csv"), index=False)
    print(f"Saved: oof_probs_{model_name}.npy, test_probs_{model_name}.npy, meta_{model_name}.csv")

# -------------------
# CLI
# -------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["convnext_base", "efficientnet_b3"], help="timm model names")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--holdout_valid", action="store_true", help="If set, do not use VALID for K-fold (keep as holdout)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    for model_name in args.models:
        run_model_kfold(model_name, args)
