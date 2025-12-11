# coreComponents/transformers/BART.py

from transformers import BartForConditionalGeneration, BartConfig as HF_BartConfig


class BartConfig(HF_BartConfig):
    """
    Thin wrapper around HuggingFace BART config.
    """
    pass


class BART(BartForConditionalGeneration):
    """
    Custom BART class (inherits HF implementation).
    """
    pass


def build_model(tokenizer):
    """
    Build a scratch-initialized BART model compatible with your tokenizer.
    Works with TransformersPlugin autodiscovery.
    """

    pad_id = tokenizer.stoi["<pad>"]
    eos_id = tokenizer.stoi["<end>"]

    config = BartConfig(
        vocab_size=tokenizer.get_vocab_size(),
        
        # BART architecture parameters (different from PEGASUS)
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
        bos_token_id=eos_id,  # BART uses eos as bos
        
        # BART-specific settings
        max_position_embeddings=1024,
        attention_dropout=0.1,
        dropout=0.1,
        activation_dropout=0.1,
        
        # BART uses a different activation function
        activation_function="gelu",
        
        # BART initialization parameters
        init_std=0.02,
        classifier_dropout=0.0,
        
        # BART scale embedding
        scale_embedding=False,
        
        # Encoder-decoder flag
        is_encoder_decoder=True,
        
        # BART uses tied word embeddings
        tie_word_embeddings=True,
    )

    return BART(config)