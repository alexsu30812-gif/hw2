"""Small synthetic throughput probe, not a trained music model or quality result."""
import argparse
import gc
import json
from pathlib import Path
import platform
import time

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, width=384, heads=6):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width), nn.Dropout(0.1))

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for a in (q, k, v)]
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.projection(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class ProbeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(600, 384)
        self.position = nn.Embedding(4096, 384)
        self.blocks = nn.Sequential(*[Block() for _ in range(6)])
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 600, bias=False)
        self.head.weight = self.embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens):
        hidden = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        logits = self.head(self.norm(self.blocks(hidden)))
        return F.cross_entropy(logits[:, :-1].reshape(-1, 600), tokens[:, 1:].reshape(-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    records = []
    for length in [1024, 2048, 4096]:
        model = ProbeDecoder().to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        tokens = torch.randint(0, 600, (1, length), device=device)
        times, losses = [], []
        for step in range(3):
            if device == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(tokens)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite synthetic loss')
            loss.backward()
            optimizer.step()
            if device == 'mps':
                torch.mps.synchronize()
            times.append(time.perf_counter() - start)
            losses.append(float(loss.detach().cpu()))
        seconds = sum(times[1:]) / 2
        item = {'sequence_length': length, 'batch_size': 1,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'warmup_seconds': times[0], 'mean_step_seconds': seconds,
                'tokens_per_second': length / seconds, 'synthetic_losses': losses}
        if device == 'mps':
            item['mps_current_allocated_bytes'] = torch.mps.current_allocated_memory()
            item['mps_driver_allocated_bytes'] = torch.mps.driver_allocated_memory()
        records.append(item)
        print(json.dumps(item), flush=True)
        del model, optimizer, tokens, loss
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    result = {'synthetic_only': True, 'music_model_trained': False,
              'device': device, 'machine': platform.machine(),
              'torch_version': torch.__version__, 'dtype': 'float32',
              'architecture': 'random GPT-style decoder implemented in torch, 6 layers, width 384, 6 heads, SDPA, tied embeddings',
              'limitations': 'Two timed optimizer steps per length after one warmup; no data loading, variable-length packing, long-run thermal behavior, convergence or generation quality measured.',
              'measurements': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
