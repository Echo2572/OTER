import torch
from torch import nn


class FullRelationModel(nn.Module):
    """Checkpoint-compatible wrapper around the fixed Full encoder."""

    def __init__(self, sentence_encoder, num_classes, rel2id):
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_classes
        self.linear = nn.Linear(sentence_encoder.hidden_size, num_classes)
        self.fc = nn.Linear(sentence_encoder.hidden_size, num_classes)
        self.drop = nn.Dropout()
        self.rel2id = rel2id
        self.id2rel = {index: relation for relation, index in rel2id.items()}

    def forward(self, *args):
        representation, auxiliary_loss = self.sentence_encoder(*args)
        logits = self.fc(self.drop(representation))
        return logits, representation, auxiliary_loss

