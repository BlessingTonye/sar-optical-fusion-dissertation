from losses import CombinedLoss
from evaluate import evaluate_macro_f1

import os
import torch
import numpy as np
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def train_config(model, config_name, train_ds, val_ds, weights_tensor, n_classes, device,
                  checkpoint_dir, max_epochs, patience, batch_size, lr, weight_decay):
    """
    Trains a segmentation model with checkpoint selection and early
    stopping based on validation macro-F1.

    Args:
        model (nn.Module): model to train.
        config_name (str): identifier used to name the checkpoint folder.
        train_ds (Dataset): training dataset.
        val_ds (Dataset): validation dataset.
        weights_tensor (Tensor): inverse-frequency class weights.
        n_classes (int): number of segmentation classes.
        device (torch.device): device to train on.
        checkpoint_dir (str): base directory for saved checkpoints.
        max_epochs (int): maximum number of training epochs.
        patience (int): epochs to wait for improvement before stopping.
        batch_size (int): training/validation batch size.
        lr (float): learning rate for AdamW.
        weight_decay (float): weight decay for AdamW.

    Returns:
        tuple: (checkpoint path, best validation macro-F1, training history dict).
    """
    model = model.to(device)
    criterion = CombinedLoss(class_weights=weights_tensor.to(device), n_classes=n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_f1 = -1
    epochs_no_improve = 0

    run_dir = f"{checkpoint_dir}/{config_name}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = f"{run_dir}/best_model.pt"

    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, lbls).item()
        avg_val_loss = val_loss / len(val_loader)

        val_f1 = evaluate_macro_f1(model, val_loader, n_classes, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_macro_f1'].append(val_f1)

        print(f"[{config_name}] Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
              f"val_loss={avg_val_loss:.4f}, val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (val_macro_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"[{config_name}] Early stopping at epoch {epoch+1}")
                break

    return ckpt_path, best_val_f1, history


def train_config_dual(config_name, model, train_ds, val_ds, weights_tensor, n_classes, device,
                       checkpoint_dir, max_epochs, patience, batch_size, lr, weight_decay):
    """
    Trains a two-input (dual-encoder) segmentation model, such as
    MiddleFusionUNet, with checkpoint selection and early stopping
    based on validation macro-F1.

    Args:
        config_name (str): identifier used to name the checkpoint folder.
        model (nn.Module): two-input model to train (expects forward(x_s1, x_s2)).
        train_ds (Dataset): training dataset, yielding (img_s1, img_s2, label) per item.
        val_ds (Dataset): validation dataset, same format as train_ds.
        weights_tensor (Tensor): inverse-frequency class weights.
        n_classes (int): number of segmentation classes.
        device (torch.device): device to train on.
        checkpoint_dir (str): base directory for saved checkpoints.
        max_epochs (int): maximum number of training epochs.
        patience (int): epochs to wait for improvement before stopping.
        batch_size (int): training/validation batch size.
        lr (float): learning rate for AdamW.
        weight_decay (float): weight decay for AdamW.

    Returns:
        tuple: (checkpoint path, best validation macro-F1, training history dict).
    """
    model = model.to(device)
    criterion = CombinedLoss(class_weights=weights_tensor.to(device), n_classes=n_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_f1 = -1
    epochs_no_improve = 0
    run_dir = f"{checkpoint_dir}/{config_name}"
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = f"{run_dir}/best_model.pt"
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for img_s1, img_s2, lbls in train_loader:
            img_s1, img_s2, lbls = img_s1.to(device), img_s2.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(img_s1, img_s2)
            loss = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for img_s1, img_s2, lbls in val_loader:
                img_s1, img_s2, lbls = img_s1.to(device), img_s2.to(device), lbls.to(device)
                logits = model(img_s1, img_s2)
                val_loss += criterion(logits, lbls).item()
                preds = logits.argmax(dim=1).cpu().numpy().flatten()
                all_preds.append(preds)
                all_targets.append(lbls.cpu().numpy().flatten())
        avg_val_loss = val_loss / len(val_loader)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        valid_mask = all_targets != -1
        val_f1 = f1_score(all_targets[valid_mask], all_preds[valid_mask], labels=list(range(n_classes)), average='macro')

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_macro_f1'].append(val_f1)

        print(f"[{config_name}] Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (val_macro_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"[{config_name}] Early stopping at epoch {epoch+1}")
                break

    return ckpt_path, best_val_f1, history