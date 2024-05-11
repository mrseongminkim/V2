import random
import os
import subprocess as sp

import torch
import torch.nn as nn
import numpy as np
import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        # pos: example_length, batch * example
        # neg: example_length, batch * example
        # regex: regex_length, batch
        pos = batch["pos"].to(device)
        neg = batch["neg"].to(device)
        regex = batch["regex"].to(device)

        output = model(pos, neg, regex, teacher_forcing_ratio)
        # output: regex_length * batch * vocab_size

        vocab_size = output.shape[-1]

        # Skip <sos> token
        output = output[1:].view(-1, vocab_size)
        # output: (regex_length - 1) * batch, vocab_size

        regex = regex[1:].view(-1)
        # regex = (regex_length - 1) * batch, vocab_size

        loss = criterion(output, regex)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            # pos: example_length, batch * example
            # neg: example_length, batch * example
            # regex: regex_length, batch
            # pos_len: batch * example
            # neg_len: batch * example
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)
            regex = batch["regex"].to(device)

            # turn off teacher forcing
            output = model(pos, neg, regex, 0)
            # output: regex_length * batch * vocab_size

            vocab_size = output.shape[-1]

            # Skip <sos> token
            output = output[1:].view(-1, vocab_size)
            # output: (regex_length - 1) * batch, vocab_size

            regex = regex[1:].view(-1)
            # regex = (regex_length - 1) * batch, vocab_size

            loss = criterion(output, regex)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=15, verbose=False, delta=0, path="checkpoint.pt"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = float("inf")
        self.early_stop = False
        self.delta = delta
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def __call__(self, model, epoch, train_loss, valid_loss, time):
        score = valid_loss
        if score >= self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), os.path.join(self.path, "model.pt"))
            with open(os.path.join(self.path, "log.txt"), "w") as f:
                f.write(f"epoch: {epoch + 1}\n")
                f.write(f"train loss: {train_loss:7.2f}\n")
                f.write(f"valid loss: {valid_loss:7.2f}\n")
                f.write(f"time: {time}\n")
            self.counter = 0


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
