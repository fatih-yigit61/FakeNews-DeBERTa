from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from configs.config import ENHANCED_DIM, HIDDEN_SIZE, MANIP_EMBED_DIM, MODEL_NAME, STYLE_FEAT_DIM, STYLE_PROJ_DIM


class OptimizedMultiTaskModel(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, dropout_rate: float = 0.1, use_style_in_fake: bool = True) -> None:
        super().__init__()
        # DeBERTa-v3 stores weights in FP16 internally — convert to FP32 to prevent
        # NaN during gradient updates. BF16 autocast still handles fast compute.
        self.encoder = AutoModel.from_pretrained(model_name).float()
        self.dropout = nn.Dropout(dropout_rate)
        self.use_style_in_fake = use_style_in_fake

        # Project stylometric features (5-d) to STYLE_PROJ_DIM (64-d).
        # Always built so checkpoint loading works regardless of ablation flag.
        self.style_proj = nn.Sequential(nn.Linear(STYLE_FEAT_DIM, STYLE_PROJ_DIM), nn.GELU())

        # fake_head input dimension depends on ablation flag:
        #   use_style_in_fake=True  → ENHANCED_DIM = CLS(768) + style_proj(64) = 832
        #   use_style_in_fake=False → HIDDEN_SIZE = CLS(768) only
        # Higher dropout (0.3) on fake_head to prevent memorizing source-specific patterns.
        # Other heads keep dropout_rate (0.1) since they don't suffer from source bias.
        fake_input_dim = ENHANCED_DIM if use_style_in_fake else HIDDEN_SIZE
        self.fake_head = nn.Sequential(
            nn.Linear(fake_input_dim, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2)
        )

        # 2-hidden-layer sentiment head: 768→512→256→3 for richer feature extraction
        self.sentiment_class_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 512), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 3),
        )
        self.sentiment_intensity_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64), nn.GELU(), nn.Dropout(dropout_rate), nn.Linear(64, 1)
        )
        # Wider manipulation feature extractor: 768→512→256→128 for more capacity
        self.manipulation_feature = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 512), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(256, MANIP_EMBED_DIM), nn.GELU(), nn.Dropout(dropout_rate),
        )
        self.manipulation_classifier = nn.Linear(MANIP_EMBED_DIM, 1)
        self.layer_norm = nn.LayerNorm(ENHANCED_DIM)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, style_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DeBERTa-v3 stores weights in FP16 internally — cast to FP32 for head compatibility
        cls_vec = self.dropout(enc_out.last_hidden_state[:, 0, :].float())
        # Always compute style_proj + enhanced (needed for embeddings output / GNN compat)
        style_proj = self.style_proj(style_feats)
        enhanced = self.layer_norm(torch.cat([cls_vec, style_proj], dim=-1))
        # Fake head: use enhanced (with style) or CLS-only based on ablation flag
        fake_input = enhanced if self.use_style_in_fake else cls_vec
        # manipulation path: 128-d intermediate is the task-specific embedding for GNN
        manip_hidden = self.manipulation_feature(cls_vec)
        return {
            "fake_logits": self.fake_head(fake_input),
            "sentiment_logits": self.sentiment_class_head(cls_vec),
            "sentiment_intensity": self.sentiment_intensity_head(cls_vec).squeeze(-1),
            "manipulation_logits": self.manipulation_classifier(manip_hidden).squeeze(-1),
            "embeddings": enhanced,                     # 832-d: kept for backward compat
            "manipulation_embedding": manip_hidden,     # 128-d: task-specific, exported to GNN
        }

    @torch.no_grad()
    def get_predictions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, style_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.forward(input_ids, attention_mask, style_feats)
        fake_probs = F.softmax(out["fake_logits"], dim=-1)
        sentiment_probs = F.softmax(out["sentiment_logits"], dim=-1)
        return {
            "fake_prob": fake_probs[:, 1],
            "fake_class": fake_probs.argmax(dim=-1),
            "sentiment_prob": sentiment_probs,
            "sentiment_class": sentiment_probs.argmax(dim=-1),
            "sentiment_intensity": torch.sigmoid(out["sentiment_intensity"]),
            "manipulation_score": torch.sigmoid(out["manipulation_logits"]),
            "manipulation_vector": out["manipulation_embedding"],
        }
