import argparse
import configparser
import time

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import set_seed, init_weights, EarlyStopping, train_fn, evaluate_fn
from vocabulary import Vocabulary
from model import Encoder, Decoder, Seq2Seq
from dataset import get_data_loader

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
    "--hidden_dim",
    dest="hidden_dim",
    type=int,
)
parser.add_argument(
    "--n_layers",
    dest="n_layers",
    type=int,
)
parser.add_argument(
    "--weight_decay",
    dest="weight_decay",
    type=float,
)
parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
)
parser.add_argument(
    "--rnn_type",
    dest="rnn_type",
)
parser.add_argument(
    "--set_transformer",
    action="store_true",
    default=False,
    dest="set_transformer",
)
parser.add_argument(
    "--gpu_idx",
    dest="gpu_idx",
)
opt = parser.parse_args()

train_file_path = opt.train_file_path
valid_file_path = opt.valid_file_path
expt_dir = opt.expt_dir
hidden_dim = opt.hidden_dim
n_layers = opt.n_layers
weight_decay = opt.weight_decay
batch_size = opt.batch_size
rnn_type = opt.rnn_type
set_transformer = opt.set_transformer
gpu_idx = opt.gpu_idx

expt_dir += f"/{rnn_type}_{hidden_dim}_{n_layers}_{set_transformer}"

device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

vocab = Vocabulary()
vocab_size = len(vocab)
pad_index = vocab.stoi["<pad>"]

usage = "train"
num_worker = 0
train_data_loader = get_data_loader(train_file_path, usage, pad_index, batch_size, num_worker, True)
valid_data_loader = get_data_loader(valid_file_path, usage, pad_index, batch_size, num_worker, False)

encoder = Encoder(vocab_size, hidden_dim, n_layers, rnn_type, set_transformer)
decoder = Decoder(vocab_size, hidden_dim, n_layers, rnn_type)
model = Seq2Seq(encoder, decoder, device).to(device)
model.apply(init_weights)

criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
clip = 0.5
scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=15)
early_stop = EarlyStopping(patience=15, path=expt_dir)

train_start = time.time()
for epoch in range(200):
    train_loss, train_string_accuracy, train_set_accuracy = train_fn(model, train_data_loader, optimizer, criterion, clip, device, vocab)
    valid_loss, valid_string_accuracy, valid_set_accuracy = evaluate_fn(model, valid_data_loader, criterion, device, vocab)
    print(f"Epoch: {epoch + 1}")
    print(f"Train Loss: {train_loss:.2f} | Train String acc: {train_string_accuracy:.2f} | Train Set acc: {train_set_accuracy:.2f}")
    print(f"Valid Loss: {valid_loss:.2f} | Valid String acc: {valid_string_accuracy:.2f} | Valid Set acc: {valid_set_accuracy:.2f}")
    early_stop(model, epoch, valid_loss, valid_string_accuracy, valid_set_accuracy, time.time() - train_start)
    if early_stop.early_stop:
        break
