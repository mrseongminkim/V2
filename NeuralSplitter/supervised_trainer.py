import logging
import os
import random
import time

import torch
import numpy as np
from torch import optim

from dataset import Vocabulary
from loss import NLLLoss
from optim import Optimizer
from visualize import visualize_loss
from evaluator import Evaluator
from checkpoint import Checkpoint
from EarlyStopping import EarlyStopping


def list_chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


class SupervisedTrainer:
    """The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """

    def __init__(
        self,
        expt_dir="experiment",
        loss=NLLLoss(),
        batch_size=64,
        random_seed=None,
        checkpoint_every=100,
        print_every=100,
        input_vocab=None,
        output_vocab=None,
        max_sequence_length=None,
    ):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.max_sequence_length = max_sequence_length
        self.loss = loss
        self.evaluator = Evaluator(
            loss=self.loss,
            batch_size=batch_size,
            input_vocab=input_vocab,
            max_sequence_length=max_sequence_length,
        )
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.match = 0
        self.match_seqnum = 0
        self.match_setnum = 0
        self.total = 0

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio):
        loss = self.loss

        decoder_outputs, decoder_hidden, other = model(input_variable)

        target_variable = target_variable.contiguous().view(-1, self.max_sequence_length)

        # Get loss
        loss.reset()

        # acc of comparing to input strings & loss calculating
        seqlist = other["sequence"]
        seqlist2 = [i.tolist() for i in seqlist]  # 각 Tensor를 list로 바꾼다.

        # tmp = (batch_size * num_examples) * max_seq_len
        tmp = torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()

        # acc of comparing to regex
        vocab = Vocabulary()

        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)  # batch_size * num_examples
            target = target_variable[:, step].to(device="cuda")

            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target)

            if step == 0:
                match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
            else:
                match_seq = torch.cat((match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1)

            non_padding = target.ne(vocab.stoi["<pad>"])

            self.match += seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()

            self.total += non_padding.sum().item()

        result = torch.logical_or(match_seq, target_variable.eq(vocab.stoi["<pad>"]).to(device="cuda"))

        self.match_seqnum += [example.all() for example in result].count(True)

        tmp = list_chunk([example.all() for example in result], 10)

        self.match_setnum += [all(example) for example in tmp].count(True)

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0
        epoch_loss_total = 0

        start_time = time.time()

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs
        step = start_step
        step_elapsed = 0

        # to track the average training loss per epoch as the model trains; epoch wise
        avg_train_losses = []
        # to track the average validtation loss per epoch as the model trains
        avg_valid_losses = []

        early_stopping = EarlyStopping(patience=15, verbose=True)

        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))
            self.match = 0
            self.match_seqnum = 0
            self.match_setnum = 0
            self.total = 0
            self.total_data_size = 0

            model.train(True)

            train_losses = []
            epoch_loss_total = 0

            for pos, label in data:
                step += 1
                step_elapsed += 1
                self.total_data_size += pos.size(0)

                loss = self._train_batch(
                    pos.to(device="cuda"),
                    self.max_sequence_length,
                    label,
                    model,
                    teacher_forcing_ratio,
                )
                train_losses.append(loss)
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = "Progress: %d%%, Train %s: %.4f" % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg,
                    )
                    log.info(log_msg)

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)

            accuracyT = self.match / self.total  # token wise
            acc_seqT = self.match_seqnum / (self.total_data_size * 10)  # example wise
            acc_setT = self.match_setnum / self.total_data_size  # regex wise
            train_log = "Train %s: %.4f, Accuracy: %.4f, Accuracy of seq: %.4f, Accuracy of set: %.4f" % (
                self.loss.name,
                epoch_loss_avg,
                accuracyT,
                acc_seqT,
                acc_setT,
            )

            if dev_data is not None:
                dev_loss, accuracy, acc_seq, acc_set = self.evaluator.evaluate(model, dev_data)
                avg_valid_losses.append(dev_loss)
                valid_log = "Dev %s: %.4f, Accuracy: %.4f, Accuracy of seq: %.4f, Accuracy of set: %.4f" % (self.loss.name, dev_loss, accuracy, acc_seq, acc_set)
                early_stopping(
                    acc_set,
                    model,
                    self.optimizer,
                    epoch,
                    step,
                    self.input_vocab,
                    self.output_vocab,
                    self.expt_dir,
                )
                self.optimizer.update(epoch_loss_avg, epoch)
                Checkpoint(
                    model=model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=step,
                    input_vocab=self.input_vocab,
                    output_vocab=self.output_vocab,
                ).save(
                    self.expt_dir + "/best_accuracy",
                    acc_seq,
                    acc_set,
                    epoch_loss_avg,
                    dev_loss,
                    time.time() - start_time,
                )
                model.train(mode=True)
            else:
                self.optimizer.update(acc_setT, epoch)

            if early_stopping.early_stop:
                print("Early Stopping")
                break

            log.info("Finished epoch %d:" % epoch)
            log.info(train_log)
            log.info(valid_log)
        return avg_train_losses, avg_valid_losses

    def train(
        self,
        model,
        data,
        num_epochs=5,
        resume=False,
        dev_data=None,
        optimizer=None,
        teacher_forcing_ratio=0,
    ):
        """Run training for a given model.
        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop("params", None)
            defaults.pop("initial_lr", None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)
            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))
        train_loss, valid_loss = self._train_epoches(
            data,
            model,
            num_epochs,
            start_epoch,
            step,
            dev_data=dev_data,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        visualize_loss(train_loss, valid_loss, self.expt_dir)
        return model
