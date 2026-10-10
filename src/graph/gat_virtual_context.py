from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool


class GATVirtualContextClassifier(nn.Module):
    """
    Flexible GAT model for edge_v4 ablation experiments.

    Supported context modes:
        none:
            Pure edge_v4 graph.
            No RXNFP, no mapped_v2 external feature, no virtual context node.

        rxnfp:
            Add one virtual context node per graph.
            The virtual node is generated from RXNFP only.

        mapped:
            Add one virtual context node per graph.
            The virtual node is generated from mapped_v2 only.

        rxnfp_mapped:
            Add one virtual context node per graph.
            The virtual node is generated from RXNFP + mapped_v2.

    Readout:
        none:
            global atom embedding
            + center atom embedding
            + center-local atom embedding

        rxnfp / mapped / rxnfp_mapped:
            global atom embedding
            + center atom embedding
            + center-local atom embedding
            + virtual context node embedding
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        mapped_dim: int = 0,
        context_mode: str = "rxnfp_mapped",
        dropout: float = 0.2,
    ):
        super().__init__()

        valid_context_modes = {
            "none",
            "rxnfp",
            "mapped",
            "rxnfp_mapped",
        }

        if context_mode not in valid_context_modes:
            raise ValueError(
                f"Invalid context_mode={context_mode}. "
                f"Expected one of: {sorted(valid_context_modes)}"
            )

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.mapped_dim = mapped_dim
        self.context_mode = context_mode
        self.use_virtual_context = context_mode != "none"

        if self.use_virtual_context and mapped_dim <= 0:
            raise ValueError(
                f"context_mode={context_mode} requires mapped_dim > 0."
            )

        self.dropout_layer = nn.Dropout(dropout)

        self.node_encoder = nn.Linear(node_dim, hidden_dim)

        if self.use_virtual_context:
            self.context_encoder = nn.Sequential(
                nn.Linear(mapped_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, hidden_dim),
            )
        else:
            self.context_encoder = None

        self.convs = nn.ModuleList(
            [
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                    concat=False,
                )
                for _ in range(num_layers)
            ]
        )

        self.act = nn.ReLU()

        # Pure graph:
        # global atom + center atom + center-local atom = 3 vectors
        #
        # Virtual-context graph:
        # global atom + center atom + center-local atom + context node = 4 vectors
        if self.use_virtual_context:
            final_dim = hidden_dim * 4
        else:
            final_dim = hidden_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _make_context_edges(
        self,
        num_atom_nodes: int,
        batch: torch.Tensor,
        num_graphs: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create context node indices and context edges.

        For each graph:
            context node <-> all atom nodes in that graph

        Returns:
            context_node_indices:
                shape [num_graphs]
            context_edge_index:
                shape [2, 2 * num_atom_nodes]
            context_edge_attr:
                shape [2 * num_atom_nodes, edge_dim]
        """
        atom_indices = torch.arange(
            num_atom_nodes,
            device=device,
            dtype=torch.long,
        )

        context_node_indices = (
            torch.arange(num_graphs, device=device, dtype=torch.long)
            + num_atom_nodes
        )

        atom_context_indices = context_node_indices[batch]

        # atom -> context
        edge_src_1 = atom_indices
        edge_dst_1 = atom_context_indices

        # context -> atom
        edge_src_2 = atom_context_indices
        edge_dst_2 = atom_indices

        context_edge_index = torch.stack(
            [
                torch.cat([edge_src_1, edge_src_2], dim=0),
                torch.cat([edge_dst_1, edge_dst_2], dim=0),
            ],
            dim=0,
        )

        context_edge_attr = torch.zeros(
            (context_edge_index.size(1), self.edge_dim),
            dtype=torch.float32,
            device=device,
        )

        # In edge_v4, the last edge feature is is_context_edge.
        context_edge_attr[:, -1] = 1.0

        return context_node_indices, context_edge_index, context_edge_attr

    def _masked_mean_pool(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        mask = mask.float().view(-1, 1)

        pooled_sum = global_add_pool(
            x * mask,
            batch,
            size=num_graphs,
        )

        pooled_count = global_add_pool(
            mask,
            batch,
            size=num_graphs,
        ).clamp(min=1.0)

        return pooled_sum / pooled_count

    def _get_context_input(
        self,
        data,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Get external reaction-level context feature.

        In the current dataset code, this is stored as data.mapped_x.
        Depending on mode, data.mapped_x can actually be:
            RXNFP only
            mapped_v2 only
            RXNFP + mapped_v2
        """
        if not hasattr(data, "mapped_x"):
            raise ValueError(
                f"context_mode={self.context_mode} requires data.mapped_x, "
                "but the dataset did not provide it."
            )

        context_input = data.mapped_x

        if context_input.dim() == 1:
            context_input = context_input.view(1, -1)

        if context_input.size(0) != num_graphs:
            raise ValueError(
                f"data.mapped_x graph count mismatch: "
                f"mapped_x.size(0)={context_input.size(0)}, "
                f"num_graphs={num_graphs}"
            )

        if context_input.size(1) != self.mapped_dim:
            raise ValueError(
                f"data.mapped_x dim mismatch: "
                f"got {context_input.size(1)}, expected {self.mapped_dim}. "
                f"context_mode={self.context_mode}"
            )

        return context_input

    def forward(self, data):
        atom_x_raw = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        device = atom_x_raw.device
        num_atom_nodes = atom_x_raw.size(0)

        if hasattr(data, "batch"):
            atom_batch = data.batch
        else:
            atom_batch = torch.zeros(
                num_atom_nodes,
                dtype=torch.long,
                device=device,
            )

        num_graphs = int(atom_batch.max().item()) + 1

        atom_x = self.node_encoder(atom_x_raw)

        if self.use_virtual_context:
            context_input = self._get_context_input(
                data=data,
                num_graphs=num_graphs,
            )

            context_x = self.context_encoder(context_input)

            (
                context_node_indices,
                context_edge_index,
                context_edge_attr,
            ) = self._make_context_edges(
                num_atom_nodes=num_atom_nodes,
                batch=atom_batch,
                num_graphs=num_graphs,
                device=device,
            )

            # Add virtual context nodes to hidden node matrix.
            x = torch.cat([atom_x, context_x], dim=0)

            # Add context edges to original graph edges.
            full_edge_index = torch.cat(
                [edge_index, context_edge_index],
                dim=1,
            )

            full_edge_attr = torch.cat(
                [edge_attr, context_edge_attr],
                dim=0,
            )

        else:
            # Pure graph ablation:
            # no external context, no virtual context node.
            x = atom_x
            full_edge_index = edge_index
            full_edge_attr = edge_attr
            context_node_indices = None

        for conv in self.convs:
            h = conv(x, full_edge_index, full_edge_attr)
            h = self.act(h)
            h = self.dropout_layer(h)
            x = x + h

        # Split atom nodes after message passing.
        atom_out = x[:num_atom_nodes]

        global_emb = global_mean_pool(
            atom_out,
            atom_batch,
            size=num_graphs,
        )

        if hasattr(data, "center_mask"):
            center_emb = self._masked_mean_pool(
                x=atom_out,
                batch=atom_batch,
                mask=data.center_mask,
                num_graphs=num_graphs,
            )
        else:
            center_emb = torch.zeros_like(global_emb)

        if hasattr(data, "center_local_mask"):
            center_local_emb = self._masked_mean_pool(
                x=atom_out,
                batch=atom_batch,
                mask=data.center_local_mask,
                num_graphs=num_graphs,
            )
        else:
            center_local_emb = torch.zeros_like(global_emb)

        graph_parts = [
            global_emb,
            center_emb,
            center_local_emb,
        ]

        if self.use_virtual_context:
            context_out = x[context_node_indices]
            graph_parts.append(context_out)

        graph_emb = torch.cat(graph_parts, dim=1)

        logits = self.classifier(graph_emb).view(-1)

        return logits