# coreComponents/tokenization.py

import re
from datasets import Dataset, DatasetDict

# ----------------------------------------------------------
# Geohash tokenizer (shared by ALL operations)
# ----------------------------------------------------------
class TrajectoryTokenizer:
    """
    Simple tokenizer that maps each geohash (or token) to an integer.
    Uses whitespace splitting.
    """

    def __init__(self, resolution=8):
        self.resolution = resolution
        self.stoi = {
            "<pad>": 0,
            "<unk>": 1,
            "<end>": 2,
            '<|endoftext|>': 3,
            'summarize:': 4
        }
        self.itos = {v: k for k, v in self.stoi.items()}
        self.next_id = len(self.stoi)

    def build_vocab(self, token_list):
        """
        token_list is a list of geohash strings.
        """
        unique = sorted(set(token_list))
        for tok in unique:
            if tok not in self.stoi:
                self.stoi[tok] = self.next_id
                self.itos[self.next_id] = tok
                self.next_id += 1

    def encode(self, text):
        tokens = text.split()
        ids = []
        for t in tokens:
            ids.append(self.stoi.get(t, self.stoi["<unk>"]))
        return ids

    def decode(self, ids):
        return " ".join(self.itos.get(i, "<unk>") for i in ids)

    def pad_token_id(self):
        return self.stoi["<pad>"]

    def eos_token_id(self):
        return self.stoi["<end>"]
    def get_vocab_size(self):
        return len(self.stoi)

# ----------------------------------------------------------
# Operation specific tokenizers
# ----------------------------------------------------------

def tokenize_summarization(df, tokenizer):
    """
    Input: original trajectory
    Output: summary trajectory
    """

    def map_fn(row):
        inp = "summarize: " + row["trajectory"]
        out = row["summary"]

        return {
            "input_ids": tokenizer.encode(inp),
            "labels": tokenizer.encode(out),
        }

    ds = Dataset.from_pandas(df[["trajectory", "summary"]])
    ds = ds.map(map_fn, remove_columns=["trajectory", "summary"])

    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


def tokenize_generation(df, tokenizer):
    """
    Input: trajectory
    Output: same trajectory (LM-style training)
    """

    def map_fn(row):
        ids = tokenizer.encode(row["trajectory"])
        return {
            "input_ids": ids,
            "labels": ids,
        }

    ds = Dataset.from_pandas(df[["trajectory"]])
    ds = ds.map(map_fn, remove_columns=["trajectory"])

    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


def tokenize_classification(df, tokenizer):
    """
    trajectory -> label
    """

    labels = sorted(df["label"].unique())
    label_map = {lab: i for i, lab in enumerate(labels)}

    def map_fn(row):
        return {
            "input_ids": tokenizer.encode(row["trajectory"]),
            "labels": label_map[row["label"]],
        }

    ds = Dataset.from_pandas(df[["trajectory", "label"]])
    ds = ds.map(map_fn, remove_columns=["trajectory", "label"])

    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


def tokenize_next_point(df, tokenizer):
    """
    input = trajectory up to last point  
    label = last point
    """

    def map_fn(row):
        pts = row["trajectory"].split()
        if len(pts) < 2:
            pts = ["<unk>", "<unk>"]

        inp = " ".join(pts[:-1])
        nxt = pts[-1]

        return {
            "input_ids": tokenizer.encode(inp),
            "labels": tokenizer.encode(nxt),
        }

    ds = Dataset.from_pandas(df[["trajectory"]])
    ds = ds.map(map_fn, remove_columns=["trajectory"])

    return DatasetDict({
        "train": ds,
        "validation": ds,
    })


# ----------------------------------------------------------
# Dispatcher used by AddOperation
# ----------------------------------------------------------

def tokenize_dataset(operation_type, df, tokenizer):

    if operation_type == "summarization":
        return tokenize_summarization(df, tokenizer)

    if operation_type == "generation":
        return tokenize_generation(df, tokenizer)

    if operation_type == "classification":
        return tokenize_classification(df, tokenizer)

    if operation_type == "next_point":
        return tokenize_next_point(df, tokenizer)

    raise ValueError(f"Unknown operation type: {operation_type}")