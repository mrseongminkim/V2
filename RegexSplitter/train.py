import argparse
import configparser
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vocabulary import Vocabulary
from dataset import get_data_loader
from model import Encoder, Attention, Decoder, Seq2Seq
from utils import set_seed, init_weights, count_parameters, train_fn, evaluate_fn, EarlyStopping

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")
set_seed(int(config["seed"]["train"]))

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_file_path",
    dest="train_file_path",
)
parser.add_argument(
    "--valid_file_path",
    dest="valid_file_path",
)
parser.add_argument(
    "--expt_dir",
    dest="expt_dir",
)
parser.add_argument(
    "--regex_max_length",
    dest="regex_max_length",
    type=int,
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
)
parser.add_argument(
    "--teacher_forcing_ratio",
    dest="teacher_forcing_ratio",
    type=float,
)
parser.add_argument(
    "--n_layers",
    dest="n_layers",
    type=int,
)
parser.add_argument(
    "--hidden_dim",
    dest="hidden_dim",
    type=int,
)
parser.add_argument(
    "--n_epochs",
    dest="n_epochs",
    type=int,
)
parser.add_argument(
    "--clip",
    dest="clip",
    type=float,
)
parser.add_argument(
    "--rnn_type",
    dest="rnn_type",
)

opt = parser.parse_args()
train_file_path = opt.train_file_path
valid_file_path = opt.valid_file_path
regex_max_length = opt.regex_max_length
batch_size = opt.batch_size
teacher_forcing_ratio = opt.teacher_forcing_ratio
n_layers = opt.n_layers
hidden_dim = opt.hidden_dim
n_epochs = opt.n_epochs
clip = opt.clip
expt_dir = opt.expt_dir + f"/gru/"
# expt_dir = opt.expt_dir + "/{}__{}__{}__{}".format(rnn_cell, hidden_size, n_layers, bi)
rnn_type = opt.rnn_type

vocab = Vocabulary()
vocab_size = len(vocab)
usage = "set2regex"
pad_index = vocab.stoi["<pad>"]
num_worker = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_data_loader = get_data_loader(train_file_path, usage, pad_index, regex_max_length, batch_size, num_worker, True)
valid_data_loader = get_data_loader(valid_file_path, usage, pad_index, regex_max_length, batch_size, num_worker, False)

encoder = Encoder(vocab_size, hidden_dim, n_layers, rnn_type)
attention = Attention(hidden_dim)
decoder = Decoder(vocab_size, hidden_dim, n_layers, rnn_type, attention)
model = Seq2Seq(encoder, decoder, device, rnn_type).to(device)
model.apply(init_weights)
print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=5)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

early_stop = EarlyStopping(patience=10, path=expt_dir)
train_start = time.time()
for epoch in range(n_epochs):
    train_loss = train_fn(
        model,
        train_data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        valid_data_loader,
        criterion,
        device,
    )
    print(f"Train Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"Valid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")
    scheduler.step(valid_loss)
    early_stop(model, epoch, train_loss, valid_loss, time.time() - train_start)
    if early_stop.early_stop:
        break
