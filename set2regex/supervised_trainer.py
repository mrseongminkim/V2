import logging
import os
import random
import time

import numpy as np
import torch
from torch import optim

from evaluator import Evaluator
from NeuralSplitter.loss import NLLLoss
from NeuralSplitter.optim import Optimizer
from NeuralSplitter.visualize import visualize_loss

from checkpoint import Checkpoint
from EarlyStopping import EarlyStopping


class SupervisedTrainer(object):
    def __init__(self, expt_dir="experiment", loss=NLLLoss(), batch_size=64, random_seed=None, checkpoint_every=100, print_every=100, input_vocab=None, output_vocab=None):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, input_vocab=input_vocab)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, pos, neg, regex, model, teacher_forcing_ratio):
        regex = regex.cuda()
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(pos, neg, regex, teacher_forcing_ratio=teacher_forcing_ratio)

        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            # len(decoder_outputs) = len(regex) - 1
            # step + 1: Skip over <sos> token
            loss.eval_batch(step_output, regex[:, step + 1])

        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        start_time = time.time()

        steps_per_epoch = len(data)
        total_steps = steps_per_epoch * n_epochs
        step = start_step
        step_elapsed = 0

        best_acc = 0

        # to track the training loss as the model trains
        train_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validtation loss per epoch as the model trains
        avg_valid_losses = []
        early_stopping = EarlyStopping(patience=15, verbose=True)

        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            model.train(True)
            for pos, neg, regex in data:
                step += 1
                step_elapsed += 1

                loss = self._train_batch(pos, neg, regex, model, teacher_forcing_ratio)

                train_losses.append(loss)
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = "Progress: %d%%, Train %s: %.4f" % (step / total_steps * 100, self.loss.name, print_loss_avg)
                    log.info(log_msg)

            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)

            # clear lists to track next epoch
            train_losses = []
            if step_elapsed == 0:
                continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f,  %.4f" % (epoch, self.loss.name, epoch_loss_avg, train_loss)

            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                avg_valid_losses.append(dev_loss)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                early_stopping(dev_loss, model, self.optimizer, epoch, step, self.input_vocab, self.output_vocab, self.expt_dir)
                self.optimizer.update(dev_loss, epoch)
                if accuracy > best_acc:
                    log.info("accuracy increased >> best_accuracy: {:.2f}, current_accuracy: {:.2f}".format(accuracy, best_acc))
                    Checkpoint(
                        model=model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        step=step,
                        input_vocab=self.input_vocab,
                        output_vocab=self.output_vocab,
                    ).save(self.expt_dir + "/best_accuracy", accuracy, epoch_loss_avg, dev_loss, time.time() - start_time)
                    best_acc = accuracy
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            if early_stopping.early_stop:
                print("Early Stopping")
                break
            log.info(log_msg)
        return avg_train_losses, avg_valid_losses

    def train(self, model, data, num_epochs=5, resume=False, dev_data=None, optimizer=None, teacher_forcing_ratio=0):
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
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
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

        train_loss, valid_loss = self._train_epoches(data, model, num_epochs, start_epoch, step, dev_data=dev_data, teacher_forcing_ratio=teacher_forcing_ratio)
        visualize_loss(train_loss, valid_loss, self.expt_dir)
        return model
