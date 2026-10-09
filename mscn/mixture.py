"""Optional single-table feature mixture; no online sampling or PCA."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mixture_options(config):
    """Return constructor options only when explicitly enabled."""
    if not config.get("use_adaptive_single_mixture", False):
        return None
    if config.get("use_single_embedding", 0) != 1:
        raise ValueError("Single mixture requires use_single_embedding=1")
    if config.get("has_unmatched_embedding", 0) != 0:
        raise ValueError("Single mixture requires has_unmatched_embedding=0")
    options = {
        "bitmap_dim": config.get("single_bitmap_dim", 1000),
        "embedding_dim": config.get("single_embedding_dim", 768),
        "pca_dim": config.get("single_pca_dim", 1536),
        "gate_hidden": config.get("single_mixture_gate_hidden", 64),
        "gate_mode": config.get("single_mixture_gate_mode", "learned"),
        "fixed_alpha": config.get("single_mixture_fixed_alpha", 0.5),
    }
    for key in ("bitmap_dim", "embedding_dim", "pca_dim", "gate_hidden"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == "pca_dim" else 1):
            raise ValueError(f"Invalid single mixture {key}: {value}")
    if options["gate_mode"] not in ("learned", "fixed"):
        raise ValueError("single_mixture_gate_mode must be learned or fixed")
    if not 0 <= options["fixed_alpha"] <= 1:
        raise ValueError("single_mixture_fixed_alpha must be in [0, 1]")
    return options


def _query_map(mapping, query_id):
    # Accept the existing integer-keyed .pt format and string-keyed JSON.
    keys = [key for key in (query_id, str(query_id)) if key in mapping]
    if len(keys) != 1:
        raise ValueError(f"Missing or ambiguous query ID {query_id}")
    return mapping[keys[0]]


def _read_bitmaps(path, tables, dim):
    result = []
    nbytes = (dim + 7) // 8
    with open(path, "rb") as stream:
        for query_id, query_tables in enumerate(tables):
            header = stream.read(4)
            if len(header) != 4 or int.from_bytes(header, "little") != len(query_tables):
                raise ValueError(f"Bitmap table count mismatch: {path}, query {query_id}")
            rows = []
            for _ in query_tables:
                raw = stream.read(nbytes)
                if len(raw) != nbytes:
                    raise ValueError(f"Truncated bitmap file: {path}")
                bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
                if bits[dim:].any():
                    raise ValueError(f"Nonzero bitmap byte padding: {path}")
                rows.append(bits[:dim].astype(np.float32))
            result.append(rows)
        if stream.read(1):
            raise ValueError(f"Extra records or incorrect bitmap width: {path}")
    return result


def load_mixture_samples(config, split, tables, table2vec):
    """Pack [table, Q bitmap/mean/PCA, R bitmap/mean/PCA]."""
    options = mixture_options(config)
    if options is None:
        raise ValueError("Mixture loader requires use_adaptive_single_mixture=true")
    dim, emb_dim, pca_dim = (options[k] for k in ("bitmap_dim", "embedding_dim", "pca_dim"))
    gate_description = options["gate_mode"]
    if gate_description == "fixed":
        gate_description += f" (alpha={options['fixed_alpha']})"
    print(f"[Single Mixture][{split}] gate={gate_description}; "
          f"bitmap_dim={dim}, embedding_dim={emb_dim}, pca_dim={pca_dim}", flush=True)
    files = {}
    for suffix in ("query_aware_bitmap_file", "random_bitmap_file",
                   "query_aware_embedding_file", "random_embedding_file"):
        key = f"{split}_{suffix}"
        print(f"[Single Mixture][{split}] {suffix}: {config.get(key) or '(not configured)'}", flush=True)
        if not config.get(key) or not Path(config[key]).is_file():
            raise ValueError(f"Missing single mixture input: {key}")
        files[suffix] = config[key]
    bitmaps = [_read_bitmaps(files[f"{source}_bitmap_file"], tables, dim)
               for source in ("query_aware", "random")]
    embeddings = [torch.load(files[f"{source}_embedding_file"], map_location="cpu", weights_only=True)
                  for source in ("query_aware", "random")]
    mappings = embeddings
    if any(not isinstance(mapping, dict) or len(mapping) != len(tables) for mapping in mappings):
        raise ValueError("Mixture inputs must contain exactly one mapping per query")

    encoded = []
    for query_id, query_tables in enumerate(tables):
        aliases = [table.split()[-1] for table in query_tables]
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"Duplicate table aliases in query {query_id}")
        maps = [_query_map(mapping, query_id) for mapping in mappings]
        if any(not isinstance(mapping, dict) or set(mapping) != set(aliases) for mapping in maps):
            raise ValueError(f"Mixture table aliases do not match query {query_id}")
        rows = []
        for j, (table, alias) in enumerate(zip(query_tables, aliases)):
            features = [table2vec[table]]
            for source in range(2):
                bitmap = bitmaps[source][query_id][j]
                vector = torch.as_tensor(maps[source][alias]).detach().cpu().numpy()
                if vector.shape != (emb_dim + 1 + pca_dim,) or not np.isfinite(vector).all():
                    raise ValueError(f"Expected [mean({emb_dim}), log_count, PCA({pca_dim})] for {alias}")
                # Zero-match records carry the offline empty-table embedding.
                mean = vector[:emb_dim]
                pca = vector[emb_dim + 1:]
                features.extend((bitmap, mean, pca))
            rows.append(np.concatenate(features).astype(np.float32))
        encoded.append(rows)
    print(f"[Single Mixture][{split}] Loaded {len(encoded)} queries, "
          f"{sum(len(rows) for rows in encoded)} table instances; "
          f"sample feature width={2 * (dim + emb_dim + pca_dim)} (excluding table one-hot).", flush=True)
    return encoded


class SingleSampleMixture(nn.Module):
    """Shared source encoders followed by one query/table gate for all features."""

    def __init__(self, table_dim, hidden_dim, bitmap_dim=1000, embedding_dim=768,
                 pca_dim=1536, gate_hidden=64, gate_mode="learned", fixed_alpha=0.5):
        super().__init__()
        self.bitmap_dim, self.embedding_dim, self.pca_dim = bitmap_dim, embedding_dim, pca_dim
        self.source_dim = bitmap_dim + embedding_dim + pca_dim
        self.input_dim = 2 * self.source_dim
        self.output_dim = 128 + embedding_dim * (2 if pca_dim else 1)
        self.gate_mode, self.fixed_alpha = gate_mode, fixed_alpha
        self.bitmap_proj = nn.Linear(bitmap_dim, 128)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        if pca_dim:
            self.pca_norm = nn.LayerNorm(pca_dim)
            if pca_dim != embedding_dim:
                self.pca_proj = nn.Linear(pca_dim, embedding_dim)
        if gate_mode == "learned":
            # Gate uses query structure only, never sample coverage or PCA validity.
            self.gate = nn.Sequential(nn.Linear(table_dim + 2 * hidden_dim, gate_hidden),
                                      nn.ReLU(), nn.Linear(gate_hidden, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def _encode(self, source, active):
        b, e = self.bitmap_dim, self.embedding_dim
        bitmap = F.leaky_relu(self.bitmap_proj(source[..., :b]), negative_slope=0.01)
        mean = self.emb_norm(source[..., b:b + e])
        parts = [bitmap, mean]
        if self.pca_dim:
            raw_pca = source[..., b + e:]
            # pca_valid = (raw_pca != 0).any(dim=-1, keepdim=True).to(source.dtype)
            pca = self.pca_norm(raw_pca)
            if self.pca_dim != e:
                pca = F.leaky_relu(self.pca_proj(pca), negative_slope=0.01)
            # parts.append(pca * pca_valid)
            parts.append(pca)
        # Mask padding after affine encoders; zero-match means retain _EMPTY.
        return torch.cat(parts, dim=-1) * active

    def forward(self, features, table_vecs, predicate_context, join_context, sample_mask):
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} mixture features, got {features.shape[-1]}")
        q, r = features[..., :self.source_dim], features[..., self.source_dim:2 * self.source_dim]
        if self.gate_mode == "learned":
            context = torch.cat((predicate_context, join_context), dim=-1)
            context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
            alpha = torch.sigmoid(self.gate(torch.cat((table_vecs, context), dim=-1)))
        else:
            alpha = features.new_full((*features.shape[:-1], 1), self.fixed_alpha)
        alpha = alpha * sample_mask
        hq = self._encode(q, sample_mask)
        hr = self._encode(r, sample_mask)
        mixed = (alpha * hq + (1 - alpha) * hr) * sample_mask
        # Nonpersistent diagnostics: no checkpoint keys or retained autograd graph.
        self.last_alpha = alpha.detach()
        # Bitmap non-emptiness is only for logging; it never controls alpha or encoding.
        cq = (q[..., :self.bitmap_dim] != 0).any(dim=-1, keepdim=True).to(features.dtype)
        cr = (r[..., :self.bitmap_dim] != 0).any(dim=-1, keepdim=True).to(features.dtype)
        both = cq * cr
        neither = (1 - cq) * (1 - cr)
        self.last_stats = torch.stack((sample_mask.sum(), (both * sample_mask).sum(),
                                       (alpha * both * sample_mask).sum(),
                                       (cq * (1 - cr) * sample_mask).sum(),
                                       (cr * (1 - cq) * sample_mask).sum(),
                                       (neither * sample_mask).sum())).detach()
        return mixed
