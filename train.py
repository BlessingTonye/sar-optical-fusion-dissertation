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
    Trains a single-input segmentation model and selects the best
    checkpoint based on validation macro-F1. Training stops early when
    validation macro-F1 does not improve for the specified patience.

    Args:
        model (nn.Module): Segmentation model to train.
        config_name (str): Name used to identify the model configuration and checkpoint folder.
        train_ds (Dataset): Dataset used for model training.
        val_ds (Dataset): Dataset used for model validation.
        weights_tensor (Tensor): Class weights used by the loss function.
        n_classes (int): Number of segmentation classes.
        device (torch.device): Device used for model training.
        checkpoint_dir (str): Directory for saving model checkpoints.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs without validation macro-F1 improvement before early stopping.
        batch_size (int): Batch size used for training and validation.
        lr (float): Learning rate used by the AdamW optimiser.
        weight_decay (float): Weight decay used by the AdamW optimiser.

    Returns:
        tuple: Path to the best checkpoint, best validation macro-F1, and training history.
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
    Trains a dual-input segmentation model for the Middle Fusion
    configuration and selects the best checkpoint based on validation
    macro-F1. Training stops early when validation macro-F1 does not
    improve for the specified patience.

    Args:
        config_name (str): Name used to identify the model configuration and checkpoint folder.
        model (nn.Module): Dual-input segmentation model to train.
        train_ds (Dataset): Training dataset containing corresponding SAR and optical inputs with their labels.
        val_ds (Dataset): Validation dataset containing corresponding SAR and optical inputs with their labels.
        weights_tensor (Tensor): Class weights used by the loss function.
        n_classes (int): Number of segmentation classes.
        device (torch.device): Device used for model training.
        checkpoint_dir (str): Directory for saving model checkpoints.
        max_epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs without validation macro-F1 improvement before early stopping.
        batch_size (int): Batch size used for training and validation.
        lr (float): Learning rate used by the AdamW optimiser.
        weight_decay (float): Weight decay used by the AdamW optimiser.

    Returns:
        tuple: Path to the best checkpoint, best validation macro-F1, and training history.
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
        for img_sar, img_optical, lbls in train_loader:
            img_sar, img_optical, lbls = img_sar.to(device), img_optical.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(img_sar, img_optical)
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
            for img_sar, img_optical, lbls in val_loader:
                img_sar, img_optical, lbls = img_sar.to(device), img_optical.to(device), lbls.to(device)
                logits = model(img_sar, img_optical)
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