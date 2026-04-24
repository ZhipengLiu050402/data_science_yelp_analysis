import torch
import torch.nn as nn
from transformers import AutoModel


class BERTCatClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        meta_dim: int,
        meta_hidden_dim: int = 64,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + meta_hidden_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_repr = outputs.last_hidden_state[:, 0, :]
        meta_repr = self.meta_mlp(meta_features)

        fused = torch.cat([cls_repr, meta_repr], dim=-1)
        fused = self.dropout(fused)

        logits = self.classifier(fused)
        return logits