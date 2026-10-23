"""Small causal music Transformer, randomly initialized; no pretrained weights."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    width: int = 384
    layers: int = 6
    heads: int = 6
    context: int = 4096
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.ln1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.out = nn.Linear(config.width, config.width)
        self.drop = nn.Dropout(config.dropout)
        self.ln2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, hidden, past=None, cache=False):
        batch, length, width = hidden.shape
        q, k, v = self.qkv(self.ln1(hidden)).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        old_length = 0
        if past is not None:
            old_length = past[0].shape[2]
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        if old_length and length > 1:
            allowed = torch.arange(k.shape[2], device=hidden.device)[None, :] <= (
                old_length + torch.arange(length, device=hidden.device)[:, None])
            attention = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        else:
            attention = F.scaled_dot_product_attention(q, k, v, is_causal=(old_length == 0))
        hidden = hidden + self.drop(self.out(attention.transpose(1, 2).reshape(batch, length, width)))
        hidden = hidden + self.mlp(self.ln2(hidden))
        return hidden, (k, v) if cache else None


class MusicTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self.initialize)
        self.head.weight = self.tokens.weight
        for block in self.blocks:
            nn.init.normal_(block.out.weight, std=0.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(2 * config.layers))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, past=None, use_cache=False, loss_weights=None):
        offset = 0 if past is None else past[0][0].shape[2]
        if ids.shape[1] + offset > self.config.context:
            raise ValueError('Context overflow; re-prefill the retained token window')
        positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
        hidden = self.dropout(self.tokens(ids) + self.positions(positions))
        cache = []
        for i, block in enumerate(self.blocks):
            hidden, kv = block(hidden, None if past is None else past[i], use_cache)
            if use_cache:
                cache.append(kv)
        logits = self.head(self.norm(hidden))
        loss = None
        if targets is not None:
            terms = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), targets.reshape(-1),
                                    ignore_index=-100, reduction='none').reshape_as(targets)
            weights = (targets != -100).to(terms.dtype)
            if loss_weights is not None:
                weights = weights * loss_weights
            loss = (terms * weights).sum() / weights.sum().clamp_min(1)
        return logits, loss, cache if use_cache else None


def save_checkpoint(path, model, metadata, optimizer=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = {'format_version': 1, 'model_config': asdict(model.config),
               'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'metadata': metadata}
    if optimizer is not None:
        content['optimizer'] = optimizer.state_dict()
    temporary = path.with_suffix('.temporary')
    torch.save(content, temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    content = torch.load(path, map_location='cpu', weights_only=True)
    if content['format_version'] != 1:
        raise ValueError('Unsupported model checkpoint')
    model = MusicTransformer(ModelConfig(**content['model_config']))
    model.load_state_dict(content['state_dict'], strict=True)
    return model.to(device), content['metadata']
