import argparse
import logging
import time
import configparser
import os
import sys

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from NeuralSplitter.seed import seed_all
from NeuralSplitter.loss import NLLLoss
from NeuralSplitter.optim import Optimizer

from dataset import get_data_loader
from models import EncoderRNN2, DecoderRNN2, Seq2seq2
from supervised_trainer import SupervisedTrainer

parser = argparse.ArgumentParser()
# data setting
parser.add_argument(
    "--train_path",
    default="./data/practical_data/integrated/test_snort.csv",
    dest="train_path",
    help="Specify the path to the training data file",
)
parser.add_argument(
    "--valid_path",
    default="./data/practical_data/integrated/test_snort.csv",
    dest="valid_path",
    help="Specify the path to the validation data file",
)
parser.add_argument(
    "--expt_dir",
    action="store",
    dest="expt_dir",
    default="./saved_models",
    help="Path to the experiment directory. If load_checkpoint is True, then path to the checkpoint directory has to be provided",
)
# hyperparameter setting
parser.add_argument(
    "--hidden_size",
    action="store",
    dest="hidden_size",
    default=256,
    type=int,
    help="Specify the size of the hidden layer.",
)
parser.add_argument(
    "--num_layer",
    action="store",
    dest="num_layer",
    default=2,
    type=int,
    help="Specify the number of layers in the model.",
)
parser.add_argument(
    "--bidirectional",
    action="store_true",
    dest="bidirectional",
    default=True,
    help="Indicate if the model is bidirectional",
)
parser.add_argument(
    "--lr",
    action="store",
    dest="lr",
    default=0.001,
    type=float,
    help="Specify the learning rate for the training process.",
)
parser.add_argument(
    "--dropout_en",
    action="store",
    dest="dropout_en",
    default=0.0,
    type=float,
    help="Specify the dropout rate for the encoder.",
)
parser.add_argument(
    "--dropout_de",
    action="store",
    dest="dropout_de",
    default=0.0,
    type=float,
    help="Specify the dropout rate for the decoder.",
)
parser.add_argument(
    "--weight_decay",
    action="store",
    dest="weight_decay",
    default=0.0,
    type=float,
    help="Specify the weight decay hyperparameter for L2 regularization.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    dest="batch_size",
    default=512,
    type=int,
    help="Specify the batch size.",
)
parser.add_argument(
    "--rnn_cell",
    action="store",
    dest="rnn_cell",
    help="Specify whether to use GRU cell for RNN. If not specified, LSTM will be used by default.",
)
parser.add_argument(
    "--set_transformer",
    action="store_true",
    default=False,
    dest="set_transformer",
    help="Specify whether to use the set transformer in encoder2.",
)
parser.add_argument(
    "--use_attn",
    action="store_true",
    dest="use_attn",
    default=True,
    help="Specify whether to use attention mechanism",
)
parser.add_argument(
    "--attn_mode",
    action="store_true",
    dest="attn_mode",
    default=False,
    help="Specify the chosen attention mode.",
)
# etc
parser.add_argument(
    "--load_checkpoint",
    action="store",
    dest="load_checkpoint",
    help="Specify the name of the checkpoint to load, typically an encoded time string.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    dest="resume",
    default=False,
    help="Indicate if training has to be resumed from the latest checkpoint",
)
parser.add_argument(
    "--log-level",
    dest="log_level",
    default="info",
    help="Specify the log level. By default, the log level is set to 'info'.",
)
parser.add_argument(
    "--gpu_idx",
    action="store",
    dest="gpu_idx",
    default="0",
    help="Specify the index of the GPU to be used.",
)
parser.add_argument(
    "--add_seed",
    action="store",
    dest="seed",
    help="Specify the seed to be added for reproducibility.",
    type=int,
    default=1,
)
opt = parser.parse_args()

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

config = configparser.ConfigParser()
config.read("config.ini", encoding="utf-8")

seed_all(int(config["seed"]["train"]) + opt.seed)

device = torch.device(f"cuda:{int(opt.gpu_idx)}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

train_path = opt.train_path
valid_path = opt.valid_path
batch_size = opt.batch_size
example_max_len = 10 if "random" in opt.train_path else 15
regex_max_len = 100

train = get_data_loader(train_path, usage="set2regex", batch_size=batch_size, shuffle=True, example_max_len=example_max_len, regex_max_len=regex_max_len)
dev = get_data_loader(valid_path, usage="set2regex", batch_size=batch_size, shuffle=False, example_max_len=example_max_len, regex_max_len=regex_max_len)

input_vocab = train.dataset.vocab
output_vocab = train.dataset.vocab
padding_index = input_vocab.stoi["<pad>"]

rnn_cell = opt.rnn_cell
loss = NLLLoss()
# loss = NLLLoss(ignore_index=padding_index)
# This causes loss to be nan when all the targets are pad
if torch.cuda.is_available():
    loss.cuda()

s2smodel = None
optimizer = None

hidden_size = opt.hidden_size
n_layers = opt.num_layer
bidirectional = opt.bidirectional
attn_mode = opt.attn_mode
bi = "2" if bidirectional else "1"
expt_dir = opt.expt_dir + f"/{rnn_cell}_{hidden_size}_{n_layers}_{attn_mode}"

set_transformer = opt.set_transformer
encoder = EncoderRNN2
decoder = DecoderRNN2

if not opt.resume:
    encoder = encoder(
        vocab_size=len(input_vocab),
        max_len=example_max_len,
        hidden_size=hidden_size,
        dropout_p=opt.dropout_en,
        input_dropout_p=0.0,
        bidirectional=bidirectional,
        n_layers=n_layers,
        rnn_cell=rnn_cell,
        variable_lengths=False,
        set_transformer=set_transformer,
    )
    decoder = decoder(
        vocab_size=len(input_vocab),
        max_len=regex_max_len,
        hidden_size=hidden_size * (2 if bidirectional else 1),
        dropout_p=opt.dropout_de,
        input_dropout_p=0.0,
        use_attention=True,
        bidirectional=bidirectional,
        rnn_cell=rnn_cell,
        n_layers=n_layers,
        attn_mode=opt.attn_mode,
    )
    s2smodel = Seq2seq2(encoder, decoder)
    if torch.cuda.is_available():
        s2smodel.cuda()
    # All the functions in this module are intended to be used to initialize neural network parameters,
    # so they all run in torch.no_grad() mode and will not be taken into account by autograd.
    for param in s2smodel.parameters():
        param.data.uniform_(-0.1, 0.1)

    optimizer = Optimizer(
        torch.optim.Adam(s2smodel.parameters(), lr=opt.lr),
        max_grad_norm=0,  # 1, 3, 5, 8, 10
    )
    scheduler = ReduceLROnPlateau(optimizer.optimizer, "min", factor=0.1, patience=10)
    optimizer.set_scheduler(scheduler)

t = SupervisedTrainer(
    loss=loss,
    batch_size=batch_size,
    checkpoint_every=1800,
    print_every=100,
    expt_dir=expt_dir,
)

start_time = time.time()
s2smodel = t.train(
    s2smodel,
    train,
    num_epochs=200,
    dev_data=dev,
    optimizer=optimizer,
    teacher_forcing_ratio=0.5,
    resume=opt.resume,
)
end_time = time.time()
print("total time > ", end_time - start_time)
