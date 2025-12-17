# coreComponents/data_collator.py

import torch
from typing import Dict, List


class TrajectoryDataCollator:
    """
    Data collator for trajectory tasks.
    Handles padding and batch preparation for different operations.
    """
    
    def __init__(self, tokenizer, max_length=256, task_type="summarization"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_type = task_type
        self.pad_token_id = tokenizer.pad_token_id()
    
    def __call__(self, features: List[Dict]) -> Dict:
        """
        Collate a batch of features.
        
        Args:
            features: List of dictionaries with 'input_ids' and 'labels'
            
        Returns:
            Batched tensors ready for model
        """
        # Separate input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences
        batch_input_ids = self._pad_sequences(input_ids, self.max_length)
        batch_labels = self._pad_sequences(labels, self.max_length, pad_value=-100)
        
        # Create attention mask
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels
        }
    
    def _pad_sequences(self, sequences: List[List[int]], max_length: int, 
                      pad_value= None) -> torch.Tensor:
        """
        Pad sequences to same length.
        """
        if pad_value is None:
            pad_value = self.pad_token_id
        
        # Truncate if necessary
        sequences = [seq[:max_length] for seq in sequences]
        
        # Get max length in batch
        batch_max_len = max(len(seq) for seq in sequences)
        batch_max_len = min(batch_max_len, max_length)
        
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            padding = [pad_value] * (batch_max_len - len(seq))
            padded_sequences.append(seq + padding)
        
        return torch.tensor(padded_sequences, dtype=torch.long)


# Optional: Specialized collators for different tasks
class SummarizationCollator(TrajectoryDataCollator):
    """Collator for summarization tasks."""
    def __init__(self, tokenizer, max_length=256):
        super().__init__(tokenizer, max_length, "summarization")
    
    def __call__(self, features):
        # For summarization, we might want different max lengths for input and output
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad input to max_length, output to max_length//2 (for summaries)
        batch_input_ids = self._pad_sequences(input_ids, self.max_length)
        batch_labels = self._pad_sequences(labels, self.max_length // 2, pad_value=-100)
        
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels
        }


class GenerationCollator(TrajectoryDataCollator):
    """Collator for generation tasks."""
    def __init__(self, tokenizer, max_length=256):
        super().__init__(tokenizer, max_length, "generation")
    
    def __call__(self, features):
        # For generation, input and labels are the same
        input_ids = [f["input_ids"] for f in features]
        
        batch_input_ids = self._pad_sequences(input_ids, self.max_length)
        batch_labels = batch_input_ids.clone()  # Same as input for LM
        
        # Mask the padding tokens in labels
        batch_labels[batch_labels == self.pad_token_id] = -100
        
        attention_mask = (batch_input_ids != self.pad_token_id).long()
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": attention_mask,
            "labels": batch_labels
        }