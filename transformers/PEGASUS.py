# coreComponents/transformers/PEGASUS.py

from transformers import PegasusForConditionalGeneration, PegasusConfig as HF_PegasusConfig


class PegasusConfig(HF_PegasusConfig):
    """
    Thin wrapper around HuggingFace Pegasus config.
    """
    pass


class PEGASUS(PegasusForConditionalGeneration):
    """
    Custom Pegasus class (inherits HF implementation).
    """
    pass


def build_model(tokenizer):
    """
    Build a scratch-initialized PEGASUS model compatible with your tokenizer.
    Works with TransformersPlugin autodiscovery.
    """

    pad_id = tokenizer.stoi["<pad>"]
    eos_id = tokenizer.stoi["<end>"]  # We'll map <end> → eos_token_id

    config = PegasusConfig(
        vocab_size=tokenizer.get_vocab_size(),

        # Architecture comparable to T5/BART configs you are using:
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,

        # Token IDs
        pad_token_id=pad_id,
        eos_token_id=eos_id,

        # Pegasus-specific settings
        max_position_embeddings=1024,
        attention_dropout=0.1,
        dropout=0.1,
        activation_dropout=0.1,

        # Encoder–decoder flag
        is_encoder_decoder=True,
    )

    return PEGASUS(config)