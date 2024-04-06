from collections import Counter

import torch

from loss import NLLLoss
from dataset import Vocabulary


def list_chunk(lst, n):
    return [lst[i : i + n] for i in range(0, len(lst), n)]


class Evaluator(object):
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
        match = 0
        match_seqnum = 0
        match_setnum = 0
        total = 0
        total_data_size = 0

        vocab = Vocabulary()

        with torch.no_grad():
            for inputs, outputs, regex in data:
                total_data_size += inputs.size(0)

                decoder_outputs, decoder_hidden, other = model(inputs.to("cuda"), None, outputs)
                tgt_variables = outputs.contiguous().view(-1, self.max_sequence_length)

                # answer_dict = [dict(Counter(l)) for l in tgt_variables.tolist()]

                seqlist = other["sequence"]
                seqlist2 = [i.tolist() for i in seqlist]
                tmp = torch.Tensor(seqlist2).transpose(0, 1).squeeze(-1).tolist()
                predict_dict = [dict(Counter(l)) for l in tmp]

                # acc of comparing to input strings & loss calculating
                for step, step_output in enumerate(decoder_outputs):
                    batch_size = tgt_variables.size(0)
                    target = tgt_variables[:, step].to(device="cuda")  # 총 10개의 스텝
                    loss.eval_batch(step_output.contiguous().view(batch_size, -1), target)

                    if step == 0:
                        match_seq = seqlist[step].view(-1).eq(target).unsqueeze(-1)
                    else:
                        match_seq = torch.cat(
                            (match_seq, seqlist[step].view(-1).eq(target).unsqueeze(-1)), dim=1
                        )

                    non_padding = target.ne(vocab.stoi["<pad>"])
                    match += (
                        seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    )
                    total += non_padding.sum().item()

                result = torch.logical_or(
                    match_seq, tgt_variables.eq(vocab.stoi["<pad>"]).to(device="cuda")
                )
                match_seqnum += [example.all() for example in result].count(True)

                tmp = list_chunk([example.all() for example in result], 10)
                match_setnum += [all(example) for example in tmp].count(True)

        acc_seq = match_seqnum / (total_data_size * 10)
        acc_set = match_setnum / total_data_size
        if total == 0:
            accuracy = float("nan")
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy, acc_seq, acc_set
