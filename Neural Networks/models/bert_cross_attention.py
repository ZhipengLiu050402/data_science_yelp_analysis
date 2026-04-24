import torch
import torch.nn as nn
from transformers import AutoModel


class BERTCrossAttentionClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_classes: int,
        meta_dim: int,
        num_meta_tokens: int = 4,
        dropout: float = 0.3,
        num_heads: int = 4,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.num_meta_tokens = num_meta_tokens
        self.hidden_size = hidden_size

        # 把 meta feature 映射成多个“meta token”
        self.meta_proj = nn.Linear(meta_dim, hidden_size * num_meta_tokens)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta_features: torch.Tensor,
    ) -> torch.Tensor:
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_tokens = bert_outputs.last_hidden_state
        # shape: [B, T, H]

        batch_size = meta_features.size(0)

        meta_tokens = self.meta_proj(meta_features)
        meta_tokens = meta_tokens.view(batch_size, self.num_meta_tokens, self.hidden_size)
        # shape: [B, M, H]

        # query=text, key/value=meta
        attn_output, _ = self.cross_attn(
            query=text_tokens,
            key=meta_tokens,
            value=meta_tokens,
            need_weights=False,
        )

        fused_tokens = self.norm1(text_tokens + attn_output)
        ffn_output = self.ffn(fused_tokens)
        fused_tokens = self.norm2(fused_tokens + ffn_output)

        cls_repr = fused_tokens[:, 0, :]
        cls_repr = self.dropout(cls_repr)

        logits = self.classifier(cls_repr)
        return logits