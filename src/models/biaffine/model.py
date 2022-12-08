from typing import Optional
from transformers import BertPreTrainedModel
import torch
from torch import nn, Tensor
from torch.nn import Embedding
from alchemy.runner import AlchemyRunner
from ..trf_enc import TrfEncoder


class BiaffineLayer(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x),out_size,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class Biaffine(BertPreTrainedModel):

    def __init__(
        self,
        config, num_types: int, biaffine_dim=128,
        dropout=0.1, pool_type="max",
        use_lstm=False, lstm_layers=3, lstm_drop=0.1,
        token_ebd: Optional[Embedding] = None,
        pos_ebd: Optional[Embedding] = None,
        char_ebd: Optional[Embedding] = None,
        char_lstm_layers=1,
        char_lstm_drop=0.2,
    ):
        super().__init__(config)

        self.encoder = TrfEncoder(
            config=config,
            ebd_drop=dropout,
            pool_type=pool_type,
            use_lstm=use_lstm, lstm_layers=lstm_layers, lstm_drop=lstm_drop,
            token_ebd=token_ebd,
            pos_ebd=pos_ebd,
            char_ebd=char_ebd, char_lstm_layers=char_lstm_layers, char_lstm_drop=char_lstm_drop
        )

        self.biaffne_layer = BiaffineLayer(biaffine_dim, num_types)

        self.start_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.hidden_size, out_features=biaffine_dim),
            torch.nn.ReLU()
        )

        self.end_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.hidden_size, out_features=biaffine_dim),
            torch.nn.ReLU()
        )

    def forward(
            self,
            encodings: Tensor,
            encoding_masks: Tensor,
            token2encoding_masks: Tensor,
            token2start: Tensor,
            token_masks: Tensor,
            pos_encoding: Optional[Tensor] = None,
            w2v_encoding: Optional[Tensor] = None,
            char_encoding: Optional[Tensor] = None,
            token_masks_char: Optional[Tensor] = None,
            char_count: Optional[Tensor] = None,
    ):
        # padding_mask = ~token_masks
        h_token = self.encoder.forward(
            encodings=encodings,
            encoding_masks=encoding_masks,
            token2encoding_masks=token2encoding_masks,
            token2start=token2start,
            token_masks=token_masks,
            pos_encoding=pos_encoding,
            w2v_encoding=w2v_encoding,
            char_encoding=char_encoding,
            token_masks_char=token_masks_char,
            char_count=char_count,
        )

        start_logits = self.start_layer.forward(h_token)
        end_logits = self.end_layer.forward(h_token)

        span_logits = self.biaffne_layer.forward(start_logits, end_logits)
        span_logits = span_logits.contiguous()

        return span_logits
