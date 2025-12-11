from transformers import T5ForConditionalGeneration, T5Config as HF_T5Config

class T5Config(HF_T5Config):
    pass

class T5(T5ForConditionalGeneration):
    pass

def build_model(tokenizer):
    """
    Construct a scratch-initialized T5 model using tokenizer tokens.
    """

    config = T5Config(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_heads=8,
        num_decoder_layers=6,
        is_encoder_decoder=True,
        pad_token_id=tokenizer.stoi["<pad>"],
        eos_token_id=tokenizer.stoi["<end>"],
        decoder_start_token_id=tokenizer.stoi["summarize:"],
    )

    return T5(config)