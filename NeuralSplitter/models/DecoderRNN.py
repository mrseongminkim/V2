import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.attention import Attention
from models.baseRNN import BaseRNN


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = "attention_score"
    KEY_LENGTH = "length"
    KEY_SEQUENCE = "sequence"

    # except sos, eos
    def __init__(
        self,
        vocab_size,
        max_len,
        hidden_size,
        n_layers=1,
        rnn_cell="LSTM",
        bidirectional=False,
        input_dropout_p=0,
        dropout_p=0,
        use_attention=False,
        attn_mode=False,
    ):
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.embed_size = 4

        # self.rnn = self.rnn_cell(hidden_size, hidden_size*2, n_layers, batch_first=True, dropout=dropout_p)
        self.rnn = self.rnn_cell(vocab_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.attn_mode = attn_mode
        self.rnn1_hidden = None
        self.init_input = None
        self.masking = None
        self.input_dropout_p = input_dropout_p
        if use_attention:
            self.attention = Attention(self.hidden_size, attn_mode)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_out1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.hidden_out2 = nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward_step(self, input_var, embedding, hidden, encoder_outputs, function):
        batch_size, set_size, seq_len = input_var.size(0), input_var.size(1), input_var.size(2)

        one_hot = F.one_hot(input_var.to(device="cuda"), num_classes=self.vocab_size)
        embedded = one_hot.view(batch_size * set_size, seq_len, -1).float()
        # embedded = embedding(input_var.reshape(batch_size * set_size, seq_len))
        embedded = self.input_dropout(embedded)

        # hidden : num_layers x batch_size x (hidden_dim * 2)
        # encoder_outputs[0] : batch_size x set_size x seq_len x hidden_dim
        # encoder_outputs[1] : batch_size x set_size x hidden_dim
        # rnn1_dim : num_layers x (batch_size * set_size) x hidden_dim

        if type(self.rnn) is nn.LSTM:
            hidden = (
                hidden[0].repeat_interleave(10, dim=1),
                hidden[1].repeat_interleave(10, dim=1),
            )  # 2, 640, 128 of tuple2
            hidden = (
                torch.cat((hidden[0], self.rnn1_hidden[0]), -1),
                torch.cat((hidden[1], self.rnn1_hidden[1]), -1),
            )  # 2, 640, 256 of tuple2
            hidden = (self.hidden_out1(hidden[0]), self.hidden_out2(hidden[1]))
        else:
            # Repeat elements of a tensor.
            hidden = hidden.repeat_interleave(10, dim=1)
            # layers, batch, 2 * hidden을 layers, batch * 10, 2 * hidden으로 바꾼다.
            # batch * example과 같이 만들어주는 것?
            hidden = torch.cat((hidden, self.rnn1_hidden), -1)

            # self.rnn1_hidden: layers * (batch * examples) * (hidden * 2)
            # 최종적으로 layers * (batch * examples) * (hidden * 2 * 2)
            hidden = self.hidden_out1(hidden)
            # Linear를 통과한 이후로 hidden * 2로 된다.

        output, hidden = self.rnn(embedded, hidden)  # (640,10,128)
        # output: (batch * examples) * max_len * hidden
        # hidden: layers * (batch * examples) * hidden

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs[0].view(batch_size * set_size, seq_len, -1))

        # 여기서 hidden_vector를 vocab size로 줄인다.
        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size * set_size, seq_len, self.output_size)

        return predicted_softmax, hidden, attn

    def forward(
        self,
        inputs=None,
        embedding=None,
        encoder_hidden=None,
        encoder_outputs=None,
        function=F.log_softmax,
        teacher_forcing_ratio=0,
        masking=None,
        rnn1_hidden=None,
    ):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # masking is not used
        if masking is not None:
            self.masking = masking

        # encoder_outputs는 (src_output, set_output)

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio)

        # inputs -> (batch, examples, max_len)
        # encoder_hidden -> (num_layer x num_dir, batch, hidden)
        decoder_hidden, rnn1_hidden = self._init_state(encoder_hidden, rnn1_hidden)
        self.rnn1_hidden = rnn1_hidden
        # decoder_hidden -> if bidirecional: (num_layer, batch, 2 x hidden) else : (num_layer x num_dir, batch, hidden)

        decoder_outputs = []
        sequence_symbols = []
        # 여기서 max_len은 num_exam이니 10, [10] * batch_size; [10, 10, 10, ..., 10]
        lengths = np.array([max_length] * batch_size)

        # step: single symbol index of regex, step_output = (640,12)
        def decode(step, step_output, step_attn):
            # decoder_outputs는 List[torch.Tensor((batch * examples) * vocab)]
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            # topk = (values, indices)
            # here indices are symbol index
            # batch * examples 만큼의 indices 즉, (batch * examples) * 1
            sequence_symbols.append(symbols)
            # 이거를 걍 리스트에 넣으니까 List[torch.Tensor, ]가 됨 -> 이거 걍 cat을 하던가 하면 안 됨? torch가 다루기 편할텐데
            return symbols

        # decoder_input = inputs[:, 0].unsqueeze(1)  # (batch, 1) # (batch, set, len) -> (batch,1,len)
        # print(inputs.shape) # input variable 64,10,10
        # print(max_length)   # 10
        # 임베딩 안 씀
        # fucntion F.log_softmax
        decoder_output, decoder_hidden, attn = self.forward_step(inputs, embedding, decoder_hidden, encoder_outputs, function=function)

        # decoder_output: (batch * examples) * max_len * hidden
        # max_length만큼 돌린다.
        for di in range(decoder_output.size(1)):
            step_output = decoder_output[:, di, :]
            if attn is not None:
                if self.attn_mode:
                    step_attn = (
                        (attn[0][0][:, di, :, :], attn[0][1][:, di, :, :]),
                        (attn[1][0][:, di, :], attn[1][1][:, di, :]),
                    )
                else:  # attn only pos
                    step_attn = attn[:, di, :]
            else:
                step_attn = None
            # 이러면 di: index, step_output:첫번재 token, 두번째 token, 세번째 token
            decode(di, step_output, step_attn)

        # 이렇게 된다면 최종적으로 decoder_outputs: max_len * (batch * num_examples) * vocab
        # sequence_symbols는 max_len * (batch * num_examples) * 1

        # 이걸 바꿀 수 있다는 것을 알자.

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols

        # [10] * batch_size [10, 10, 10, ..., 10]
        # num_examples인 것 같은데...
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        # decoder_outputs으로 loss를 계산하러 간다.

        # 하고 싶은 것
        # decoder_outputs가 batch * num_examples * max_len * vocab
        # sequence_symbols가 batch * num_examples * 1

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden, sub_hidden):
        """Initialize the encoder hidden state."""
        if encoder_hidden is None:
            return None
        if type(self.rnn) == nn.LSTM:
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
            sub_hidden = tuple([self._cat_directions(h) for h in sub_hidden])
        else:
            # cat_direction: direction을 concat해준다. encoder_hidden(set)이 decoder hidden이 되고, rnn1_hidden은 rnn1 hidden 그대로
            encoder_hidden = self._cat_directions(encoder_hidden)
            sub_hidden = self._cat_directions(sub_hidden)

        return encoder_hidden, sub_hidden

    def _cat_directions(self, h):
        """If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """

        if self.bidirectional_encoder:
            h = torch.cat([h[0 : h.size(0) : 2], h[1 : h.size(0) : 2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if type(self.rnn_cell) is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif type(self.rnn_cell) is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            # 이건 examples의 size인데... 왜 max_len을 examples로 하는 것인가?
            max_length = inputs.size(1)  # minus the start of sequence symbol

        # 한 마디로 그냥 반환함
        return inputs, batch_size, max_length
