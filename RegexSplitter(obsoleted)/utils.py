import random
import os
import subprocess as sp

import torch
import torch.nn as nn
import numpy as np
import tqdm

from vocabulary import Vocabulary


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


def check_match(output: torch.Tensor, label: torch.Tensor, vocab: Vocabulary):
    with torch.no_grad():
        batch = output.shape[1] // 10
        example = 10
        example_length = output.shape[0]

        # output: example_length, batch * example, vocab_size
        output = output.argmax(dim=-1)
        # output: example_length, batch * example

        output = output.permute(1, 0).view(batch, example, example_length)
        # batch, example, example_length

        # label: batch, example * example_length
        label = label.permute(1, 0).view(batch, example, example_length)
        # label: batch, example, example_lengths

        pad_mask = label.eq(vocab.stoi["<pad>"])
        match = output.eq(label).logical_or(pad_mask)

        matched_string = torch.all(match, dim=-1).sum().item()

        match = match.reshape(batch, example * example_length)

        matched_set = torch.all(match, dim=1).sum().item()

        return matched_string, matched_set


def train_fn(model, data_loader, optimizer, criterion, clip, device, vocab):
    model.train()
    epoch_loss = 0
    n_string = 0
    string_match = 0
    set_match = 0
    for i, batch in enumerate(tqdm.tqdm(data_loader)):
        # pos: example_length, batch * example
        # label: example_length, batch * example
        pos = batch["pos"].to(device)
        label = batch["label"].to(device)

        example_length = label.shape[0]

        n_string += label.shape[-1]

        output = model(pos)
        # output: example_length, batch * example, vocab_size

        vocab_size = output.shape[-1]
        output = output.view(-1, vocab_size)
        label = label.view(-1)

        loss = criterion(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        matched_string, matched_set = check_match(output.view(example_length, -1, vocab_size), label.view(example_length, -1), vocab)
        string_match += matched_string
        set_match += matched_set
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
    loss = epoch_loss / len(data_loader)
    string_accuracy = string_match / n_string
    set_accuracy = set_match / (n_string // 10)
    return loss, string_accuracy, set_accuracy


def evaluate_fn(model, data_loader, criterion, device, vocab):
    model.eval()
    epoch_loss = 0
    n_string = 0
    string_match = 0
    set_match = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(data_loader)):
            # pos: example_length, batch * example
            # label: example_length, batch * example
            pos = batch["pos"].to(device)
            label = batch["label"].to(device)

            n_string += label.shape[-1]

            output = model(pos)
            # output: example_length, batch * example, vocab_size

            matched_string, matched_set = check_match(output, label, vocab)
            string_match += matched_string
            set_match += matched_set

            vocab_size = output.shape[-1]
            output = output.view(-1, vocab_size)
            label = label.view(-1)

            loss = criterion(output, label)
            epoch_loss += loss.item()
    loss = epoch_loss / len(data_loader)
    string_accuracy = string_match / n_string
    set_accuracy = set_match / (n_string // 10)
    return loss, string_accuracy, set_accuracy


class EarlyStopping:
    def __init__(self, patience=15, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.counter = 0
        self.best_score = -float("inf")
        self.early_stop = False
        self.delta = delta
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

    def __call__(self, model, epoch, valid_loss, valid_string_accuracy, valid_set_accuracy, time):
        score = valid_set_accuracy
        if score <= self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), os.path.join(self.path, "model.pt"))
            with open(os.path.join(self.path, "log.txt"), "w") as f:
                f.write(f"epoch: {epoch + 1}\n")
                f.write(f"valid loss: {valid_loss:.2f}\n")
                f.write(f"valid string acc: {valid_string_accuracy:.2f}\n")
                f.write(f"valid set acc: {valid_set_accuracy:.2f}\n")
                f.write(f"time: {time}\n")
            self.counter = 0


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def synthesize_regex(pos, neg, model, vocab, device, regex_max_length):
    model.eval()
    with torch.no_grad():
        # if the given pos and neg are list of strings
        # we need to assert len is 10
        if isinstance(pos, list) and isinstance(neg, list):
            pass
        encoder_outputs, hidden, cell = model.encoder(pos, neg)
        inputs = vocab.lookup_indices(["<sos>"])
        for _ in range(regex_max_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell, attention = model.decoder(inputs_tensor, encoder_outputs, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == vocab.stoi["<eos>"]:
                break
        tokens = vocab.lookup_tokens(inputs)
    regex = "".join(tokens[1:-1])
    return regex
