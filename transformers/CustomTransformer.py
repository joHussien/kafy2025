import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput


class CustomTransformerConfig(PretrainedConfig):
    model_type = "youssef_transformer"

    def __init__(
        self,
        vocab_size=32000,
        d_model=256,
        num_layers=4,
        nhead=8,
        dim_feedforward=1024,
        pad_token_id=0,
        eos_token_id=1,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward


class CustomTransformer(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config):
        super().__init__(config)

        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # Simple Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=config.num_layers
        )

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        x = self.embedding(input_ids)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        hidden = self.transformer(x)

        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits
        )
def build_model(tokenizer):
    """
    Construct a scratch-initialized T5 model using tokenizer tokens.
    """

    config = CustomTransformerConfig(
        vocab_size=tokenizer.get_vocab_size()
    )

    return CustomTransformer(config)