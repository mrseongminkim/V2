import torch
import torchtext

from NeuralSplitter.loss import NLLLoss


class Evaluator(object):
    def __init__(self, loss=NLLLoss(), batch_size=64, input_vocab=None):
        self.loss = loss
        self.batch_size = batch_size
        self.input_vocab = input_vocab

    def evaluate(self, model, data):
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = torch.device("cuda") if torch.cuda.is_available() else -1
        with torch.no_grad():
            for pos, neg, regex in data:
                regex = regex.cuda()
                decoder_outputs, decoder_hidden, other = model(pos, neg, regex)

                seqlist = other["sequence"]
                for step, step_output in enumerate(decoder_outputs):
                    target = regex[:, step + 1]
                    loss.eval_batch(step_output, target)

                    non_padding = target.ne(1)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()

        if total == 0:
            accuracy = float("nan")
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
