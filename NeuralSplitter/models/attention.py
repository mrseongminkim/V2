import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, dim, attn_mode):
        super(Attention, self).__init__()
        self.mask = None
        self.attn_mode = attn_mode  # True (attention both pos and neg) # False (attention only pos samples)
        # dim = hidden_dim * 2
        self.linear_out = nn.Linear(dim * 3, dim)
        self.dim = dim

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
        self.mask = self.mask.view(batch_by_n_examples, example_max_len).unsqueeze(1)
        example_context = context[0].view(batch_by_n_examples, example_max_len, hidden_size)
        set_context = context[1].repeat_interleave(10, dim=0)

        example_attention = self.example_attention(output, example_context)
        set_attention = self.set_attention(output, set_context)

        example_context = torch.bmm(example_attention, example_context)
        set_context = torch.bmm(set_attention, set_context)

        all_the_information = torch.cat((output, example_context, set_context), dim=-1)
        output = torch.tanh(self.linear_out(all_the_information))

        attn = (example_attention, set_attention)

        return output, attn
