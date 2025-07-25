TRANSFORMERS = {}

def load(transformer_name):
    if transformer_name == "GPT":
        from transformers import GPT2LMHeadModel
        return GPT2LMHeadModel.from_pretrained("gpt2")
    elif transformer_name == "BERT":
        from transformers import BertForSequenceClassification
        return BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # Add others as needed

def register(name, model):
    TRANSFORMERS[name] = model

def get(name):
    return TRANSFORMERS.get(name)

def exists(name):
    return name in TRANSFORMERS