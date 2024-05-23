import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, attn_mode):
        super(Attention, self).__init__()
        self.mask = None
        self.attn_mode = attn_mode  # True (attention both pos and neg) # False (attention only pos samples)
        # dim = hidden_dim * 2
        self.linear_out = nn.Linear(dim * 2, dim)
        self.dim = dim

        self.set_linear = nn.Linear(dim * 2, dim)

    def set_mask(self, mask):
        self.mask = mask

    def example_attention(self, output, example_context):
        attn = torch.bmm(output, example_context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill(self.mask, -float("inf"))
        attn = torch.softmax(attn, dim=-1)
        return attn

    def set_attention(self, output, set_attention):
        attn = torch.bmm(output, set_attention.transpose(1, 2))
        attn = torch.softmax(attn, dim=-1)
        return attn

    def forward(self, output, context):
        batch_by_n_examples, example_max_len, hidden_size = output.shape
        batch, n_examples = self.mask.size(0), self.mask.size(1)

        set_output = context[1].repeat_interleave(10, dim=0)
        set_attention = self.set_attention(output, set_output)
        set_context = torch.bmm(set_attention, set_output)
        output_with_set_info = torch.cat((output, set_context), dim=-1)
        output = torch.tanh(self.set_linear(output_with_set_info))

        output = output.view(batch, n_examples * example_max_len, hidden_size)
        self.mask = self.mask.view(batch, n_examples * example_max_len).unsqueeze(-1)

        example_output = context[0].reshape(batch, n_examples * example_max_len, hidden_size)
        example_attention = self.example_attention(output, example_output)
        example_context = torch.bmm(example_attention, example_output)
        output_with_example_info = torch.cat((output, example_context), dim=-1)
        output = torch.tanh(self.linear_out(output_with_example_info))

        attn = (example_attention, set_attention)

        return output, attn
