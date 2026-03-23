import json
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from configs.config import MANIP_EMBED_DIM

if TYPE_CHECKING:
    from src.training.text_trainer import Model1ExpertTrainer


class GNNFeatureExporter:
    def __init__(self, trainer: "Model1ExpertTrainer", upfd_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
        self.trainer = trainer
        self.upfd_dir = Path(upfd_dir or trainer.cfg.upfd_dir)
        self.output_dir = Path(output_dir or trainer.cfg.gnn_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_upfd_news(self) -> Dict[str, str]:
        news_dict: Dict[str, str] = {}
        json_path = self.upfd_dir / "news_content.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                raw = json.load(f)
            for news_id, content in raw.items():
                if isinstance(content, dict):
                    title = content.get("title", "")
                    text = content.get("text", content.get("body", ""))
                    full_text = f"{title} </s> {text}".strip(" </s>") if title else text
                else:
                    full_text = str(content)
                news_dict[str(news_id)] = full_text
        return news_dict

    def export(self, overwrite: bool = False) -> str:
        news_dict = self._load_upfd_news()
        news_ids = list(news_dict.keys())
        index: Dict[str, str] = {}
        for i in tqdm(range(0, len(news_ids), self.trainer.cfg.gnn_batch_size), desc="GNN feature extraction"):
            batch_ids = news_ids[i : i + self.trainer.cfg.gnn_batch_size]
            batch_texts = [news_dict[nid] for nid in batch_ids]
            batch_out = self.trainer.predict_batch(batch_texts)
            for j, nid in enumerate(batch_ids):
                pt_path = self.output_dir / f"{nid}.pt"
                if pt_path.exists() and not overwrite:
                    continue
                feat = {
                    "news_id": nid,
                    "manipulation_vector": batch_out["manipulation_vectors"][j],
                    "fake_score": batch_out["fake_scores"][j].item(),
                    "sentiment_class": batch_out["sentiment_classes"][j].item(),
                    "sentiment_intensity": batch_out["sentiment_intensities"][j].item(),
                    "manipulation_score": batch_out["manipulation_scores"][j].item(),
                }
                torch.save(feat, pt_path)
                index[nid] = str(pt_path)
        index_path = self.output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        if index:
            vecs = [torch.load(p, map_location="cpu", weights_only=False)["manipulation_vector"] for p in index.values()]
            torch.save({"news_ids": list(index.keys()), "vectors": torch.stack(vecs, dim=0), "dim": MANIP_EMBED_DIM}, self.output_dir / "feature_matrix.pt")
        return str(index_path)
