import time
import logging

import torch
import hydra
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from matplotlib import pyplot as plt

from .data import get_batch
from .models import TransformerLM
from .optimizer import AdamW, get_lr_cosine_schedule
from .nn_utils import cross_entropy, gradient_clipping
from .serialization import save_checkpoint, load_checkpoint

log = logging.getLogger(__name__)

def train(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        lr,
        warmup_ratio,
        betas,
        eps,
        weight_decay,
        data_path,
        val_data_path,
        batch_size,
        steps,
        max_norm,
        save_path,
        val_steps = 20,
        eval_interval = 100,
    ):

    # wandb.init(
    #     project="SLM pretrain test",
    #     config=locals()
    # )
    # wandb.login(key=["local-fa29fadb80423956e6d1b7a557cff4b0a8adaf42"])
    t0 = time.time()

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay)

    wandb.watch(model, log="all", log_freq=100)

    datas = np.load(data_path, mmap_mode="r")
    val_datas = np.load(val_data_path, mmap_mode="r")

    # datas = np.memmap(data_path, dtype=np.uint16, mode="r")
    # val_datas = np.memmap(val_data_path, dtype=np.uint16, mode="r")

    warmup_iters = int(warmup_ratio * steps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for i in tqdm(range(steps)):
        data, target = get_batch(datas, batch_size, context_length, device)

        data = data.to(torch.long)
        target = target.to(torch.long)

        # assert data.max() < vocab_size, f"Data tensor contains token ID {data.max().item()} which is out of bounds (>= {vocab_size})"
        # assert target.max() < vocab_size, f"Target tensor contains token ID {target.max().item()} which is out of bounds (>= {vocab_size})" 

        current_lr = get_lr_cosine_schedule(iter=i, max_learning_rate=lr, min_learning_rate=0.01 * lr, warmup_iters=warmup_iters, cosine_cycle_iters=steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)
        log.info(f"At step {i}, loss is {loss.cpu().item()}")
        loss.backward()
        gradient_norm = gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        if i % eval_interval == 0 or i == steps - 1:

            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(val_steps):
                    val_data, val_target = get_batch(val_datas, batch_size, context_length, device)
                    val_data, val_target = val_data.to(torch.long), val_target.to(torch.long)

                    val_logits = model(val_data)
                    val_loss = cross_entropy(val_logits, val_target)
                    val_losses.append(val_loss)
            avg_val_loss = sum(val_losses) / len(val_losses)

            elapsed_time = time.time() - t0

            wandb.log({
                "losses/train": loss.item(),
                "losses/valid": avg_val_loss,
                "learning rate": current_lr,
                "wallclock time secs": elapsed_time
            }, step=i)

        else: 
            elapsed_time = time.time() - t0
            wandb.log({
                "losses/train": loss.item(),
                "learning rate": current_lr, 
                "gradient norm": gradient_norm,
                "wallclock time secs": elapsed_time
            }, step=i)

        if i % 1000 == 0:
            save_checkpoint(model, optimizer, i, save_path)
    
    t1 = time.time()
    wandb.finish()
    print(f"total cost {t1 - t0}s.")

def hydra_train(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        data_path: str,
        val_data_path: str,
        batch_size: int,
        context_length: int,
        steps: int,
        warmup_ratio: float,
        val_steps: int,
        eval_interval: int,
        save_path: str,
        max_norm: float,
        device: str,
    ):

    t0 = time.time()

    wandb.watch(model, log="all", log_freq=100)

    datas = np.load(data_path, mmap_mode="r")
    val_datas = np.load(val_data_path, mmap_mode="r")

    warmup_iters = int(warmup_ratio * steps)

    lr = optimizer.param_groups[0]['lr']

    model.to(device)
    for i in tqdm(range(steps)):
        data, target = get_batch(datas, batch_size, context_length, device)

        data = data.to(torch.long)
        target = target.to(torch.long)

        current_lr = get_lr_cosine_schedule(iter=i, max_learning_rate=lr, min_learning_rate=0.01 * lr, warmup_iters=warmup_iters, cosine_cycle_iters=steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)
        log.info(f"At step {i}, loss is {loss.cpu().item()}")
        loss.backward()
        gradient_norm = gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        if i % eval_interval == 0 or i == steps - 1:

            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(val_steps):
                    val_data, val_target = get_batch(val_datas, batch_size, context_length, device)
                    val_data, val_target = val_data.to(torch.long), val_target.to(torch.long)

                    val_logits = model(val_data)
                    val_loss = cross_entropy(val_logits, val_target)
                    val_losses.append(val_loss)
            avg_val_loss = sum(val_losses) / len(val_losses)

            elapsed_time = time.time() - t0

            wandb.log({
                "losses/train": loss.item(),
                "losses/valid": avg_val_loss,
                "learning rate": current_lr,
                "wallclock time secs": elapsed_time
            }, step=i)

        else: 
            elapsed_time = time.time() - t0
            wandb.log({
                "losses/train": loss.item(),
                "learning rate": current_lr, 
                "gradient norm": gradient_norm,
                "wallclock time secs": elapsed_time
            }, step=i)

        if i % 1000 == 0:
            save_checkpoint(model, optimizer, i, save_path)
    
    t1 = time.time()
    wandb.finish()
    print(f"total cost {t1 - t0}s.")

def train_overfit_one_batch(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        lr,
        warmup_ratio,
        betas,
        eps,
        weight_decay,
        data_path,
        val_data_path,
        batch_size,
        steps,
        max_norm,
        save_path,
        val_steps = 20,
        eval_interval = 100,
    ):
    t0 = time.time()

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay=0.0)

    wandb.watch(model, log="all", log_freq=100)

    datas = np.load(data_path, mmap_mode="r")
    # val_datas = np.load(val_data_path, mmap_mode="r")

    warmup_iters = int(warmup_ratio * steps)
    cosine_cycle_iters = steps - warmup_iters

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    data, target = get_batch(datas, batch_size, context_length, device)

    data = data.to(torch.long)
    target = target.to(torch.long)

    model.train()
    for i in tqdm(range(steps)):
        # current_lr = get_lr_cosine_schedule(iter=i, max_learning_rate=lr, min_learning_rate=1e-6, warmup_iters=warmup_iters, cosine_cycle_iters=cosine_cycle_iters)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = current_lr

        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)
        print(f"At step {i}, loss is ", loss.cpu().item())
        loss.backward()
        # gradient_norm = gradient_clipping(model.parameters(), max_norm)
        gradient_norm = check_gradient(model.parameters())

        optimizer.step()

        elapsed_time = time.time() - t0
        wandb.log({
            "losses/train": loss.item(),
            "learning rate": lr, 
            "gradient norm": gradient_norm,
            "wallclock time secs": elapsed_time
        }, step=i)

        if i % 1000 == 0:
            save_checkpoint(model, optimizer, i, save_path)
    
    t1 = time.time()
    wandb.finish()
    print(f"total cost {t1 - t0}s.")

def lr_range_test(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        rope_theta,
        lr,
        betas,
        eps,
        weight_decay,
        data_path,
        val_data_path,
        batch_size,
        steps,
        max_norm,
        save_path,
        val_steps = 20,
        eval_interval = 100,
    ):

    t0 = time.time()

    model = TransformerLM(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, rope_theta)
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay)

    datas = np.load(data_path, mmap_mode="r")
    val_datas = np.load(val_data_path, mmap_mode="r")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    lrs_to_plot = []
    losses_to_plot = []

    start_lr = 1e-8
    end_lr = 1
    num_steps = 250

    lr_multiplier = (end_lr / start_lr) ** (1.0 / num_steps)
    current_lr = start_lr

    for i in tqdm(range(num_steps)):
        data, target = get_batch(datas, batch_size, context_length, device)

        data = data.to(torch.long)
        target = target.to(torch.long)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)

        if torch.isnan(loss) or loss.item() > min(losses_to_plot, default=1e9) * 4:
            print(f"Loss exploded at step {i}, Stopping test.")
            break

        loss.backward()
        # gradient_norm = gradient_clipping(model.parameters(), max_norm)

        optimizer.step()

        lrs_to_plot.append(current_lr)
        losses_to_plot.append(loss.item())

        current_lr *= lr_multiplier
    t1 = time.time()
    
    print(f"total cost {t1 - t0}s.")
    # 4. 绘制结果
    plt.figure(figsize=(10, 6))

    # 关键：X轴必须是对数尺度 (log scale)！
    plt.plot(lrs_to_plot, losses_to_plot)
    plt.xscale('log') 
    plt.xlabel('Learning Rate (Log Scale)')
    plt.ylabel('Loss')
    plt.title('LR Range Test')
    plt.grid(True, which="both", ls="--")

    # (可选) 找到 loss 最小值的 LR
    min_loss = min(losses_to_plot)
    min_loss_idx = losses_to_plot.index(min_loss)
    min_loss_lr = lrs_to_plot[min_loss_idx]
    print(f"Minimum loss {min_loss:.4f} found at LR = {min_loss_lr:.2e}")

    # 绘制一个红点标记最低点
    plt.plot(min_loss_lr, min_loss, 'ro') 

    plt.savefig("lr-range-test-256-250.png")
    plt.show()
    