from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.graph.reaction_graph_edge_v4 import (
    EDGE_V4_EDGE_DIM,
    EDGE_V4_NODE_DIM,
    build_reaction_graph as build_edge_v4_graph,
)
from src.graph.reaction_graph_edge_v3 import (
    build_reaction_graph as build_edge_v3_graph,
)
from src.graph.reaction_graph_edge_v2 import (
    build_reaction_graph as build_edge_v2_graph,
)


EDGE_V3_NODE_DIM = 39
EDGE_V3_EDGE_DIM = 19

EDGE_V2_NODE_DIM = 17
EDGE_V2_EDGE_DIM = 14

MAX_FALLBACK_WARNINGS = 20
_fallback_warning_count = 0


EDGE_V4_STAT_KEYS = [
    "num_shared_maps",
    "num_changed_mapping_edges",
    "num_1hop_mapping_edges",
    "num_2hop_mapping_edges",
    "num_far_candidates",
    "num_far_kept",
    "num_far_dropped",
    "far_keep_ratio",
    "informative_far_top_k",
]


def _set_default_edge_v4_stats(data: Data) -> Data:
    for key in EDGE_V4_STAT_KEYS:
        if not hasattr(data, key):
            setattr(
                data,
                key,
                torch.tensor([0.0], dtype=torch.float32),
            )
    return data


def validate_edge_v4_dims(data: Data) -> Data:
    if data.x.size(1) != EDGE_V4_NODE_DIM:
        raise ValueError(
            f"edge_v4 node_dim mismatch: got {data.x.size(1)}, "
            f"expected {EDGE_V4_NODE_DIM}"
        )

    if data.edge_attr.size(1) != EDGE_V4_EDGE_DIM:
        raise ValueError(
            f"edge_v4 edge_dim mismatch: got {data.edge_attr.size(1)}, "
            f"expected {EDGE_V4_EDGE_DIM}"
        )

    if not hasattr(data, "center_mask"):
        data.center_mask = torch.zeros(data.x.size(0), dtype=torch.float32)

    if not hasattr(data, "center_local_mask"):
        data.center_local_mask = data.center_mask.clone().float()

    data = _set_default_edge_v4_stats(data)

    return data


def pad_graph_to_edge_v4_dims(data: Data) -> Data:
    node_dim = data.x.size(1)
    edge_dim = data.edge_attr.size(1)

    if node_dim > EDGE_V4_NODE_DIM:
        raise ValueError(
            f"Cannot pad graph to edge_v4 node dim: got node_dim={node_dim}, "
            f"target={EDGE_V4_NODE_DIM}"
        )

    if edge_dim > EDGE_V4_EDGE_DIM:
        raise ValueError(
            f"Cannot pad graph to edge_v4 edge dim: got edge_dim={edge_dim}, "
            f"target={EDGE_V4_EDGE_DIM}"
        )

    node_pad = EDGE_V4_NODE_DIM - node_dim
    if node_pad > 0:
        data.x = torch.cat(
            [
                data.x,
                torch.zeros(
                    data.x.size(0),
                    node_pad,
                    dtype=data.x.dtype,
                    device=data.x.device,
                ),
            ],
            dim=1,
        )

    edge_pad = EDGE_V4_EDGE_DIM - edge_dim
    if edge_pad > 0:
        data.edge_attr = torch.cat(
            [
                data.edge_attr,
                torch.zeros(
                    data.edge_attr.size(0),
                    edge_pad,
                    dtype=data.edge_attr.dtype,
                    device=data.edge_attr.device,
                ),
            ],
            dim=1,
        )

    if not hasattr(data, "center_mask"):
        data.center_mask = torch.zeros(data.x.size(0), dtype=torch.float32)

    if not hasattr(data, "center_local_mask"):
        data.center_local_mask = data.center_mask.clone().float()

    data = _set_default_edge_v4_stats(data)

    return validate_edge_v4_dims(data)


def make_dummy_edge_v4_graph(y: int) -> Data:
    x = torch.zeros((1, EDGE_V4_NODE_DIM), dtype=torch.float32)

    edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long,
    )

    edge_attr = torch.zeros((1, EDGE_V4_EDGE_DIM), dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )

    data.center_mask = torch.zeros(1, dtype=torch.float32)
    data.center_local_mask = torch.zeros(1, dtype=torch.float32)

    data.y = torch.tensor([int(y)], dtype=torch.float32)

    data = _set_default_edge_v4_stats(data)

    return data


class ReactionGraphEdgeV4Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        reaction_col: str = "MappedReaction",
        mapped_features: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.reaction_col = reaction_col
        self.mapped_features = mapped_features

        if self.mapped_features is not None:
            if len(self.mapped_features) != len(self.df):
                raise ValueError(
                    f"mapped_features length ({len(self.mapped_features)}) "
                    f"must match dataframe length ({len(self.df)})"
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        global _fallback_warning_count

        row = self.df.iloc[idx]
        mapped_reaction = row[self.reaction_col]
        y = int(row[self.label_col])

        used_fallback = False
        fallback_type = "none"

        try:
            data = build_edge_v4_graph(
                mapped_reaction=mapped_reaction,
                y=y,
            )
            data = validate_edge_v4_dims(data)

        except Exception as edge_v4_error:
            used_fallback = True

            try:
                data = build_edge_v3_graph(
                    mapped_reaction=mapped_reaction,
                    y=y,
                )
                data = pad_graph_to_edge_v4_dims(data)
                fallback_type = "edge_v3_padded"

                if _fallback_warning_count < MAX_FALLBACK_WARNINGS:
                    print(
                        f"\n[EdgeV4 fallback warning] idx={idx}, "
                        f"fallback_type={fallback_type}, "
                        f"edge_v4_error={repr(edge_v4_error)}, "
                        f"reaction={str(mapped_reaction)[:300]}"
                    )
                    _fallback_warning_count += 1

            except Exception as edge_v3_error:
                try:
                    data = build_edge_v2_graph(
                        mapped_reaction=mapped_reaction,
                        y=y,
                    )
                    data = pad_graph_to_edge_v4_dims(data)
                    fallback_type = "edge_v2_padded"

                    if _fallback_warning_count < MAX_FALLBACK_WARNINGS:
                        print(
                            f"\n[EdgeV4 fallback warning] idx={idx}, "
                            f"fallback_type={fallback_type}, "
                            f"edge_v4_error={repr(edge_v4_error)}, "
                            f"edge_v3_error={repr(edge_v3_error)}, "
                            f"reaction={str(mapped_reaction)[:300]}"
                        )
                        _fallback_warning_count += 1

                except Exception as edge_v2_error:
                    data = make_dummy_edge_v4_graph(y=y)
                    fallback_type = "dummy"

                    if _fallback_warning_count < MAX_FALLBACK_WARNINGS:
                        print(
                            f"\n[EdgeV4 fallback dummy warning] idx={idx}, "
                            f"edge_v4_error={repr(edge_v4_error)}, "
                            f"edge_v3_error={repr(edge_v3_error)}, "
                            f"edge_v2_error={repr(edge_v2_error)}, "
                            f"reaction={str(mapped_reaction)[:300]}"
                        )
                        _fallback_warning_count += 1

        data.used_fallback = torch.tensor(
            [float(used_fallback)],
            dtype=torch.float32,
        )

        if fallback_type == "none":
            fallback_id = 0
        elif fallback_type == "edge_v3_padded":
            fallback_id = 1
        elif fallback_type == "edge_v2_padded":
            fallback_id = 2
        else:
            fallback_id = 3

        data.fallback_type = torch.tensor(
            [fallback_id],
            dtype=torch.long,
        )

        if self.mapped_features is not None:
            data.mapped_x = torch.tensor(
                self.mapped_features[idx],
                dtype=torch.float32,
            ).view(1, -1)

        return data