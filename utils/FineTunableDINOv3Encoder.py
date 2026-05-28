from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

class FineTunableDINOv3Encoder(nn.Module):
    """
    DINOv3 backbone with the last N transformer blocks unfrozen and a
    BNNeck-style projection head on top (BatchNorm -> Linear -> BatchNorm)
    
    sources:
    - https://arxiv.org/abs/1903.07071
    - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
    - https://github.com/facebookresearch/dinov3
    - https://claude.ai/ (extraordinary for debugging)
    - google AI overview
    """

    def __init__(self,
                 model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
                 embed_dim: int = 512,
                 unfreeze_last_n_blocks: int = 2,
                 batch_size: int = 32,
                 device: str = "cuda") -> None:
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone_dim = self.backbone.config.hidden_size   # 1024 for ViT-L
        self.embed_dim = embed_dim
        self.unfreeze_n = unfreeze_last_n_blocks
        self.batch_size = batch_size
        self.device = device
        self.feat_dim = embed_dim
        self.name = "FineTunable DINOv3"

        # bag of tricks paper's projection head
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.backbone_dim),
            nn.Linear(self.backbone_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
        )
        self._set_freezing(unfreeze_last_n_blocks)
        self.to(device)

    def _transformer_blocks(self):
        return self.backbone.model.layer

    def _final_norm(self):
        return self.backbone.norm

    def _set_freezing(self, unfreeze_last_n_blocks: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        blocks = self._transformer_blocks()
        n = max(0, min(unfreeze_last_n_blocks, len(blocks))) # can't unfreeze more than the available blocks
        for blk in blocks[-n:] if n > 0 else []:
            for p in blk.parameters():
                p.requires_grad_(True)
        fn = self._final_norm()
        if fn is not None and n > 0:
            for p in fn.parameters():
                p.requires_grad_(True)

    def trainable_parameter_groups(self, lr_backbone: float, lr_head: float):
        # to list only trainable parameters to the optimizer, with different lr for bb and bn head
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups

    def _pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        # mean over patch tokens (last_hidden_state[:, 0, :] == pooler_output == cls token)
        # patch level features are way richer in details for comparison (outperforms cls token)
        # this pictures explains clearly why:
        # https://github.com/facebookresearch/dinov3#overview
        return last_hidden_state[:, 1:, :].mean(dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Training forward: returns raw [B, embed_dim] embeddings."""
        out = self.backbone(pixel_values=pixel_values)
        pooled = self._pool(out.last_hidden_state)
        return self.head(pooled)

    @torch.no_grad()
    def encode_images(self, images: list, disable_progress: bool = True) -> np.ndarray:
        """l2-normalized [N, embed_dim] visual features"""
        self.eval()
        all_feats = []
        for i in tqdm(range(0, len(images), self.batch_size),
                      desc="Visual features (queries)", unit="batch",
                      disable=disable_progress):
            batch = images[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            out = self.backbone(**inputs)
            pooled = self._pool(out.last_hidden_state)
            feats = self.head(pooled).cpu().float().numpy()
            all_feats.append(feats)
        feats = np.concatenate(all_feats, axis=0)
        norms = np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-8)
        return (feats / norms).astype(np.float32)

    def encode_crops_from_paths(self, paths: list, desc: str = "FineTunable DINOv3", **kwargs) -> list:
        images = [Image.open(p).convert("RGB")
                  for p in tqdm(paths, desc=f"Loading ({desc})")]
        feats = self.encode_images(images, **kwargs)
        return [feats[i] for i in range(len(feats))]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "head": self.head.state_dict(),
            "backbone": {k: v for k, v in self.backbone.state_dict().items()
                         if self._param_was_trained(k)},
            "config": {
                "embed_dim": self.embed_dim,
                "unfreeze_last_n_blocks": self.unfreeze_n,
                "backbone_dim": self.backbone_dim,
            },
        }
        torch.save(state, path)

    def _param_was_trained(self, key: str) -> bool:
        # helps identify trained weights for a compact checkpoint
        n = self.unfreeze_n
        prefixes = []
        if n > 0:
            blocks = self._transformer_blocks()
            prefixes = [f"model.layer.{idx}."
                        for idx in range(len(blocks) - n, len(blocks))]
            prefixes.append(f"norm.")
        return any(key.startswith(p) for p in prefixes)

    def load(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location=self.device)
        self.head.load_state_dict(state["head"])

        # reapply freezing just to get the same model
        self.unfreeze_n = state["config"].get("unfreeze_last_n_blocks", self.unfreeze_n)
        self._set_freezing(self.unfreeze_n)

        # the checkpoint must contain the fine-tuned transformer blocks 
        # not just the final norm. it was not the case before :/
        saved_keys = list(state["backbone"].keys())
        n_block_keys = sum(k.startswith("model.layer") for k in saved_keys)
        if self.unfreeze_n > 0 and n_block_keys == 0:
            raise RuntimeError(
                f"Checkpoint {path} contains no transformer-block weights"
            )
        
        # https://discuss.pytorch.org/t/missing-keys-unexpected-keys-in-state-dict-when-loading-self-trained-model/22379
        missing, unexpected = self.backbone.load_state_dict(
            state["backbone"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"load: unexpected backbone keys: {unexpected}")
        if strict and missing:
            raise RuntimeError(f"load mismatch: missing={missing}")
