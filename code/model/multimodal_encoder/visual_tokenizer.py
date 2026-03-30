# from ovis

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiglipVisualTokenizer(nn.Module):
    def __init__(self,
                 vocab_size=131072,
                 tau=1.0,
                 depths=None,
                 use_indicators=True,
                 drop_cls_token=True,
                 tokenize_function="softmax",
                 hidden_stride: int = 1,
                 hidden_size: int = 1152, # hidden_size of siglip
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenize_function = tokenize_function
        self.tau = tau
        self.depths = depths
        self.use_indicators = use_indicators
        self.drop_cls_token = drop_cls_token
        self.hidden_size = hidden_size

        head_dim = vocab_size
        if self.use_indicators:
            head_dim -= 2  # reserved for two image indicator tokens

        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * hidden_stride * hidden_stride, head_dim, bias=False),
            torch.nn.LayerNorm(head_dim)
        )

        self.vte = nn.Embedding(vocab_size, hidden_size)
        nn.init.normal_(self.vte.weight, mean=0., std=1.)

    def forward(self, features):
        logits = self.head(features)
        tokens = self.tokenize(logits)

        if self.use_indicators:
            # tokens' shape is [BatchSize, #Token, VocabSize-2], so padding with [BatchSize, #Token, 2], after
            # which, tokens' shape should become [BatchSize, #Token, VocabSize]
            batch_size, token_len, _ = tokens.shape
            padding_tensor = torch.zeros(size=(batch_size, token_len, 2),
                                         dtype=tokens.dtype,
                                         device=tokens.device,
                                         layout=tokens.layout,
                                         requires_grad=False)
            tokens = torch.cat((tokens, padding_tensor), dim=2)

            # adding indicator tokens, after which tokens' shape should become [BatchSize, 1+#Token+1, VocabSize]
            begin_indicator = torch.zeros(size=(batch_size, 1),
                                          dtype=torch.long,
                                          device=tokens.device,
                                          requires_grad=False) + self.vocab_size - 2
            begin_indicator_token = F.one_hot(
                begin_indicator, num_classes=self.vocab_size).to(dtype=tokens.dtype)
            end_indicator = torch.zeros(size=(batch_size, 1),
                                        dtype=torch.long,
                                        device=tokens.device,
                                        requires_grad=False) + self.vocab_size - 1
            end_indicator_token = F.one_hot(
                end_indicator, num_classes=self.vocab_size).to(dtype=tokens.dtype)
            tokens = torch.cat((begin_indicator_token, tokens, end_indicator_token), dim=1)

        v_embed = self.vte(tokens)
        return tokens, v_embed

    def tokenize(self, logits):
        if self.tokenize_function == 'softmax':
            tokens = F.softmax(logits, dim=-1)
        elif self.tokenize_function == 'gumbel_argmax':
            tokens = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        elif self.tokenize_function == 'st_argmax':
            tokens = self.st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                f'Invalid `max_type`, expected softmax or gumbel_argmax or st_argmax, '
                f'but got {self.tokenize_function}')
        return tokens

    @staticmethod
    def st_argmax(y_soft, dim):  # straight-through softmax
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret
