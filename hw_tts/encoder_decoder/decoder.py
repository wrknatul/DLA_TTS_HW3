import torch.nn as nn

from hw_tts.transformer import FFTBlock
from .attention_masks import *


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, decoder_n_layer, encoder_dim, PAD,
                 encoder_conv1d_filter_size, encoder_head, fft_conv1d_kernel,
                 fft_conv1d_padding, dropout=0.1):

        super().__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer
        self.PAD = PAD

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.PAD)
        non_pad_mask = get_non_pad_mask(enc_pos, self.PAD)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
