import torch
import torch.nn as nn
from transformers import AutoModel


class BERTGateFusionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        meta_dim: int,
        fusion_dim: int = 256,
        dropout: float = 0.3,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        bert_hidden = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.text_proj = nn.Linear(bert_hidden, fusion_dim)
        self.meta_proj = nn.Linear(meta_dim, fusion_dim)

        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, num_classes)

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

        text_repr = self.text_proj(cls_repr)
        meta_repr = self.meta_proj(meta_features)

        gate = self.gate(torch.cat([text_repr, meta_repr], dim=-1))
        fused = gate * text_repr + (1.0 - gate) * meta_repr

        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return logits