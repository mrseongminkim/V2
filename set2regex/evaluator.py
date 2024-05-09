from collections import Counter

import torch

from loss import NLLLoss
from dataset import Vocabulary


class Evaluator:
    """Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64, input_vocab=None, max_sequence_length=10):
        self.loss = loss
        self.batch_size = batch_size
        self.input_vocab = input_vocab
        self.max_sequence_length = max_sequence_length

    def evaluate(self, model, data):
        """Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        total = 0
        match = 0

        vocab = Vocabulary()

        with torch.no_grad():
            for pos, neg, regex in data:
                decoder_outputs, decoder_hidden, other = model(pos=pos.to("cuda"), neg=neg.to("cuda"), target_variable=regex.to("cuda"))
                seqlist = other["sequence"]
                for step, step_output in enumerate(decoder_outputs):
                    step_output = step_output.to("cpu")

                    batch_size = regex.size(0)
                    target = regex[:, step + 1]

                    loss.eval_batch(step_output.contiguous().view(batch_size, -1).cpu(), target)

                    non_padding = target.ne(vocab.stoi["<pad>"])
                    correct = seqlist[step].view(-1).cpu().eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float("nan")
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
