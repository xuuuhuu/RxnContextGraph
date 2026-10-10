from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_LIST = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]

HYBRIDIZATION_LIST = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]

CHIRAL_TAG_LIST = [
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_OTHER,
]


def split_side(side_smiles: str) -> list[str]:
    if not isinstance(side_smiles, str) or not side_smiles:
        return []
    return [s for s in side_smiles.split(".") if s]


def bond_order_value(bond_type: Chem.BondType | None) -> float:
    if bond_type == Chem.BondType.SINGLE:
        return 1.0
    if bond_type == Chem.BondType.DOUBLE:
        return 2.0
    if bond_type == Chem.BondType.TRIPLE:
        return 3.0
    if bond_type == Chem.BondType.AROMATIC:
        return 1.5
    return 0.0


def bond_type_onehot_from_order(order: float) -> list[float]:
    return [
        float(order == 1.0),
        float(order == 2.0),
        float(order == 3.0),
        float(order == 1.5),
    ]


def atom_signature(atom: Chem.Atom) -> tuple:
    neighbors = []

    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        neighbors.append(
            (
                other.GetAtomMapNum(),
                other.GetAtomicNum(),
                str(bond.GetBondType()),
            )
        )

    return (
        atom.GetAtomicNum(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetChiralTag().name,
        tuple(sorted(neighbors)),
    )


def collect_mapped_info(
    smiles_block: str,
) -> tuple[
    dict[int, tuple],
    dict[tuple[int, int], Chem.BondType],
    dict[int, dict[str, float]],
    dict[int, set[int]],
]:
    """
    Collect atom-level and bond-level mapped information.

    Returns:
        atom_info:
            map number -> atom signature
        bond_info:
            (map1, map2) -> RDKit bond type
        atom_context:
            map number -> basic atom context
        adjacency:
            map number -> neighboring map numbers
    """
    atom_info: dict[int, tuple] = {}
    bond_info: dict[tuple[int, int], Chem.BondType] = {}
    atom_context: dict[int, dict[str, float]] = {}
    adjacency: dict[int, set[int]] = {}

    for smi in split_side(smiles_block):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()

            if amap <= 0:
                continue

            atom_info[amap] = atom_signature(atom)
            adjacency.setdefault(amap, set())

            atom_context[amap] = {
                "degree": float(atom.GetDegree()),
                "formal_charge": float(atom.GetFormalCharge()),
                "is_aromatic": float(atom.GetIsAromatic()),
                "total_num_hs": float(atom.GetTotalNumHs()),
                "is_in_ring": float(atom.IsInRing()),
            }

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()

            if a1 > 0 and a2 > 0:
                key = tuple(sorted((a1, a2)))
                bond_info[key] = bond.GetBondType()

                adjacency.setdefault(a1, set()).add(a2)
                adjacency.setdefault(a2, set()).add(a1)

    return atom_info, bond_info, atom_context, adjacency


def get_reaction_center_maps(
    reactant_atom_info: dict[int, tuple],
    product_atom_info: dict[int, tuple],
    reactant_bond_info: dict[tuple[int, int], Chem.BondType],
    product_bond_info: dict[tuple[int, int], Chem.BondType],
) -> tuple[
    set[int],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    all_maps = set(reactant_atom_info) | set(product_atom_info)

    changed_maps = {
        amap
        for amap in all_maps
        if reactant_atom_info.get(amap) != product_atom_info.get(amap)
    }

    all_bonds = set(reactant_bond_info) | set(product_bond_info)

    formed_bonds = {
        b
        for b in all_bonds
        if b not in reactant_bond_info and b in product_bond_info
    }

    broken_bonds = {
        b
        for b in all_bonds
        if b in reactant_bond_info and b not in product_bond_info
    }

    changed_bonds = {
        b
        for b in all_bonds
        if (
            b in reactant_bond_info
            and b in product_bond_info
            and reactant_bond_info[b] != product_bond_info[b]
        )
    }

    reaction_center_bonds = formed_bonds | broken_bonds | changed_bonds

    for bond_key in reaction_center_bonds:
        changed_maps.update(bond_key)

    return (
        changed_maps,
        reaction_center_bonds,
        formed_bonds,
        broken_bonds,
        changed_bonds,
    )


def merge_adjacency(
    reactant_adj: dict[int, set[int]],
    product_adj: dict[int, set[int]],
) -> dict[int, set[int]]:
    merged: dict[int, set[int]] = {}

    for source in [reactant_adj, product_adj]:
        for amap, neighbors in source.items():
            merged.setdefault(amap, set()).update(neighbors)

    return merged


def get_shell_maps(
    changed_maps: set[int],
    adjacency: dict[int, set[int]],
) -> tuple[set[int], set[int]]:
    """
    center-shell positional encoding:
        1-hop: direct neighbors of reaction center atoms
        2-hop: neighbors of 1-hop atoms

    Center atoms themselves are excluded from 1-hop and 2-hop.
    """
    one_hop = set()

    for amap in changed_maps:
        one_hop.update(adjacency.get(amap, set()))

    one_hop -= changed_maps

    two_hop = set()

    for amap in one_hop:
        two_hop.update(adjacency.get(amap, set()))

    two_hop -= changed_maps
    two_hop -= one_hop

    return one_hop, two_hop


def count_incident_changes(
    amap: int,
    formed_bonds: set[tuple[int, int]],
    broken_bonds: set[tuple[int, int]],
    changed_bonds: set[tuple[int, int]],
) -> dict[str, float]:
    def incident_count(bonds: set[tuple[int, int]]) -> int:
        return sum(1 for a, b in bonds if amap == a or amap == b)

    num_formed = incident_count(formed_bonds)
    num_broken = incident_count(broken_bonds)
    num_changed = incident_count(changed_bonds)

    return {
        "num_formed": float(num_formed),
        "num_broken": float(num_broken),
        "num_changed": float(num_changed),
        "is_formed_endpoint": float(num_formed > 0),
        "is_broken_endpoint": float(num_broken > 0),
        "is_changed_endpoint": float(num_changed > 0),
    }


def atom_features_edge_v3(
    atom: Chem.Atom,
    side_flag: int,
    is_center: bool,
    is_1hop_from_center: bool,
    is_2hop_from_center: bool,
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
    formed_bonds: set[tuple[int, int]],
    broken_bonds: set[tuple[int, int]],
    changed_bonds: set[tuple[int, int]],
) -> list[float]:
    """
    Node feature v3.

    Base edge_v2 node features:
        17 dims

    Added basic atom states:
        hybridization one-hot: 5
        chiral tag one-hot: 4
        total_num_hs / 4: 1
        is_in_ring: 1
        subtotal: 11

    Added node-level reaction-change features:
        degree_before
        degree_after
        degree_delta
        num_formed_incident
        num_broken_incident
        num_changed_incident
        is_formed_endpoint
        is_broken_endpoint
        is_changed_endpoint
        subtotal: 9

    Added center-shell positional encoding:
        is_1hop_from_center
        is_2hop_from_center
        subtotal: 2

    Total:
        17 + 11 + 9 + 2 = 39 dims
    """
    symbol = atom.GetSymbol()

    atom_onehot = [1.0 if symbol == a else 0.0 for a in ATOM_LIST]
    other = 0.0 if symbol in ATOM_LIST else 1.0

    base_features = atom_onehot + [
        other,
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetDegree()) / 6.0,
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(side_flag),
        float(is_center),
    ]

    hybridization = atom.GetHybridization()
    hybridization_onehot = [
        float(hybridization == h) for h in HYBRIDIZATION_LIST
    ]

    chiral_tag = atom.GetChiralTag()
    chiral_onehot = [
        float(chiral_tag == c) for c in CHIRAL_TAG_LIST
    ]

    basic_state_features = [
        *hybridization_onehot,
        *chiral_onehot,
        float(atom.GetTotalNumHs()) / 4.0,
        float(atom.IsInRing()),
    ]

    amap = atom.GetAtomMapNum()

    if amap > 0:
        before_degree = reactant_context.get(amap, {}).get("degree", 0.0)
        after_degree = product_context.get(amap, {}).get("degree", 0.0)
        degree_delta = after_degree - before_degree

        incident = count_incident_changes(
            amap=amap,
            formed_bonds=formed_bonds,
            broken_bonds=broken_bonds,
            changed_bonds=changed_bonds,
        )

        node_change_features = [
            before_degree / 6.0,
            after_degree / 6.0,
            degree_delta / 6.0,
            incident["num_formed"] / 4.0,
            incident["num_broken"] / 4.0,
            incident["num_changed"] / 4.0,
            incident["is_formed_endpoint"],
            incident["is_broken_endpoint"],
            incident["is_changed_endpoint"],
        ]
    else:
        node_change_features = [0.0] * 9

    shell_features = [
        float(is_1hop_from_center),
        float(is_2hop_from_center),
    ]

    return [
        *base_features,
        *basic_state_features,
        *node_change_features,
        *shell_features,
    ]


def bond_features_edge_v2(
    bond_order_before: float = 0.0,
    bond_order_after: float = 0.0,
    is_reactant_edge: bool = False,
    is_product_edge: bool = False,
    is_reaction_edge: bool = False,
    is_center_edge: bool = False,
    is_formed_bond: bool = False,
    is_broken_bond: bool = False,
    is_order_changed: bool = False,
) -> list[float]:
    """
    Original edge_v2 feature:
        14 dims
    """
    if is_reactant_edge:
        bond_onehot = bond_type_onehot_from_order(bond_order_before)
    elif is_product_edge:
        bond_onehot = bond_type_onehot_from_order(bond_order_after)
    else:
        bond_onehot = [0.0, 0.0, 0.0, 0.0]

    bond_order_delta = bond_order_after - bond_order_before

    return [
        *bond_onehot,
        float(is_reactant_edge),
        float(is_product_edge),
        float(is_reaction_edge),
        float(is_center_edge),
        float(is_formed_bond),
        float(is_broken_bond),
        float(is_order_changed),
        float(bond_order_before),
        float(bond_order_after),
        float(bond_order_delta),
    ]


def bond_features_edge_v3(
    bond_order_before: float = 0.0,
    bond_order_after: float = 0.0,
    is_reactant_edge: bool = False,
    is_product_edge: bool = False,
    is_reaction_edge: bool = False,
    is_center_edge: bool = False,
    is_formed_bond: bool = False,
    is_broken_bond: bool = False,
    is_order_changed: bool = False,
    edge_importance_weight: float = 0.5,
    is_changed_mapping_edge: bool = False,
    is_1hop_mapping_edge: bool = False,
    is_2hop_mapping_edge: bool = False,
    is_unchanged_mapping_edge: bool = False,
) -> list[float]:
    """
    Edge feature v3.

    Original edge_v2:
        14 dims

    Added reaction-edge weighting and typed mapping-edge indicators:
        edge_importance_weight
        is_changed_mapping_edge
        is_1hop_mapping_edge
        is_2hop_mapping_edge
        is_unchanged_mapping_edge

    Added:
        5 dims

    Total:
        14 + 5 = 19 dims
    """
    base = bond_features_edge_v2(
        bond_order_before=bond_order_before,
        bond_order_after=bond_order_after,
        is_reactant_edge=is_reactant_edge,
        is_product_edge=is_product_edge,
        is_reaction_edge=is_reaction_edge,
        is_center_edge=is_center_edge,
        is_formed_bond=is_formed_bond,
        is_broken_bond=is_broken_bond,
        is_order_changed=is_order_changed,
    )

    return [
        *base,
        float(edge_importance_weight),
        float(is_changed_mapping_edge),
        float(is_1hop_mapping_edge),
        float(is_2hop_mapping_edge),
        float(is_unchanged_mapping_edge),
    ]


def chemical_edge_weight(
    is_center_edge: bool,
    is_formed_bond: bool,
    is_broken_bond: bool,
    is_order_changed: bool,
) -> float:
    """
    Weight for normal chemical edges.

    These weights are not directly multiplied into attention.
    They are appended to edge_attr and learned by GATConv.
    """
    if is_formed_bond or is_broken_bond:
        return 1.00
    if is_order_changed:
        return 0.90
    if is_center_edge:
        return 0.80
    return 0.50


def reaction_edge_weight_and_type(
    amap: int,
    changed_maps: set[int],
    one_hop_maps: set[int],
    two_hop_maps: set[int],
) -> tuple[float, bool, bool, bool, bool]:
    """
    Complex reaction-edge weighting.

    changed mapping edge:
        reactant atom and product atom correspond to a reaction-center atom.

    1-hop mapping edge:
        atom is directly adjacent to reaction center.

    2-hop mapping edge:
        atom is two bonds away from reaction center.

    unchanged mapping edge:
        far unchanged atom mapping.
    """
    if amap in changed_maps:
        return 1.00, True, False, False, False

    if amap in one_hop_maps:
        return 0.70, False, True, False, False

    if amap in two_hop_maps:
        return 0.40, False, False, True, False

    return 0.15, False, False, False, True


def build_reaction_graph(
    mapped_reaction: str,
    y: int | None = None,
) -> Data:
    if (
        not isinstance(mapped_reaction, str)
        or ">>" not in mapped_reaction
    ):
        raise ValueError("Invalid mapped reaction.")

    reactant_block, product_block = mapped_reaction.split(">>", 1)

    (
        reactant_atom_info,
        reactant_bond_info,
        reactant_context,
        reactant_adj,
    ) = collect_mapped_info(reactant_block)

    (
        product_atom_info,
        product_bond_info,
        product_context,
        product_adj,
    ) = collect_mapped_info(product_block)

    (
        changed_maps,
        reaction_center_bonds,
        formed_bonds,
        broken_bonds,
        changed_bonds,
    ) = get_reaction_center_maps(
        reactant_atom_info,
        product_atom_info,
        reactant_bond_info,
        product_bond_info,
    )

    merged_adj = merge_adjacency(
        reactant_adj=reactant_adj,
        product_adj=product_adj,
    )

    one_hop_maps, two_hop_maps = get_shell_maps(
        changed_maps=changed_maps,
        adjacency=merged_adj,
    )

    x = []
    edge_index = []
    edge_attr = []
    center_mask = []
    center_local_mask = []

    reactant_map_to_node = {}
    product_map_to_node = {}

    node_offset = 0

    for side_flag, block, map_dict in [
        (0, reactant_block, reactant_map_to_node),
        (1, product_block, product_map_to_node),
    ]:
        is_reactant_side = side_flag == 0
        is_product_side = side_flag == 1

        for smi in split_side(block):
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                continue

            local_to_global = {}

            for atom in mol.GetAtoms():
                global_idx = node_offset
                local_to_global[atom.GetIdx()] = global_idx

                amap = atom.GetAtomMapNum()

                is_center = amap in changed_maps if amap > 0 else False
                is_1hop = amap in one_hop_maps if amap > 0 else False
                is_2hop = amap in two_hop_maps if amap > 0 else False

                x.append(
                    atom_features_edge_v3(
                        atom=atom,
                        side_flag=side_flag,
                        is_center=is_center,
                        is_1hop_from_center=is_1hop,
                        is_2hop_from_center=is_2hop,
                        reactant_context=reactant_context,
                        product_context=product_context,
                        formed_bonds=formed_bonds,
                        broken_bonds=broken_bonds,
                        changed_bonds=changed_bonds,
                    )
                )

                center_mask.append(float(is_center))
                center_local_mask.append(float(is_center or is_1hop or is_2hop))

                if amap > 0:
                    map_dict[amap] = global_idx

                node_offset += 1

            for bond in mol.GetBonds():
                i = local_to_global[bond.GetBeginAtomIdx()]
                j = local_to_global[bond.GetEndAtomIdx()]

                a1 = bond.GetBeginAtom().GetAtomMapNum()
                a2 = bond.GetEndAtom().GetAtomMapNum()

                bond_key = (
                    tuple(sorted((a1, a2)))
                    if a1 > 0 and a2 > 0
                    else None
                )

                is_center_edge = (
                    bond_key in reaction_center_bonds
                    if bond_key
                    else False
                )

                is_formed_bond = (
                    bond_key in formed_bonds
                    if bond_key
                    else False
                )

                is_broken_bond = (
                    bond_key in broken_bonds
                    if bond_key
                    else False
                )

                is_order_changed = (
                    bond_key in changed_bonds
                    if bond_key
                    else False
                )

                if bond_key is not None:
                    bond_order_before = bond_order_value(
                        reactant_bond_info.get(bond_key)
                    )
                    bond_order_after = bond_order_value(
                        product_bond_info.get(bond_key)
                    )
                else:
                    bond_order_before = 0.0
                    bond_order_after = 0.0

                weight = chemical_edge_weight(
                    is_center_edge=is_center_edge,
                    is_formed_bond=is_formed_bond,
                    is_broken_bond=is_broken_bond,
                    is_order_changed=is_order_changed,
                )

                bf = bond_features_edge_v3(
                    bond_order_before=bond_order_before,
                    bond_order_after=bond_order_after,
                    is_reactant_edge=is_reactant_side,
                    is_product_edge=is_product_side,
                    is_reaction_edge=False,
                    is_center_edge=is_center_edge,
                    is_formed_bond=is_formed_bond,
                    is_broken_bond=is_broken_bond,
                    is_order_changed=is_order_changed,
                    edge_importance_weight=weight,
                    is_changed_mapping_edge=False,
                    is_1hop_mapping_edge=False,
                    is_2hop_mapping_edge=False,
                    is_unchanged_mapping_edge=False,
                )

                edge_index.append([i, j])
                edge_attr.append(bf)

                edge_index.append([j, i])
                edge_attr.append(bf)

    shared_maps = (
        set(reactant_map_to_node.keys())
        & set(product_map_to_node.keys())
    )

    for amap in shared_maps:
        i = reactant_map_to_node[amap]
        j = product_map_to_node[amap]

        is_center = amap in changed_maps

        (
            weight,
            is_changed_mapping_edge,
            is_1hop_mapping_edge,
            is_2hop_mapping_edge,
            is_unchanged_mapping_edge,
        ) = reaction_edge_weight_and_type(
            amap=amap,
            changed_maps=changed_maps,
            one_hop_maps=one_hop_maps,
            two_hop_maps=two_hop_maps,
        )

        rf = bond_features_edge_v3(
            bond_order_before=0.0,
            bond_order_after=0.0,
            is_reactant_edge=False,
            is_product_edge=False,
            is_reaction_edge=True,
            is_center_edge=is_center,
            is_formed_bond=False,
            is_broken_bond=False,
            is_order_changed=False,
            edge_importance_weight=weight,
            is_changed_mapping_edge=is_changed_mapping_edge,
            is_1hop_mapping_edge=is_1hop_mapping_edge,
            is_2hop_mapping_edge=is_2hop_mapping_edge,
            is_unchanged_mapping_edge=is_unchanged_mapping_edge,
        )

        edge_index.append([i, j])
        edge_attr.append(rf)

        edge_index.append([j, i])
        edge_attr.append(rf)

    if len(x) == 0:
        raise ValueError("Empty reaction graph.")

    data = Data(
        x=torch.tensor(
            np.asarray(x),
            dtype=torch.float32,
        ),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long,
        ).t().contiguous(),
        edge_attr=torch.tensor(
            np.asarray(edge_attr),
            dtype=torch.float32,
        ),
    )

    data.center_mask = torch.tensor(
        center_mask,
        dtype=torch.float32,
    )

    data.center_local_mask = torch.tensor(
        center_local_mask,
        dtype=torch.float32,
    )

    if y is not None:
        data.y = torch.tensor(
            [int(y)],
            dtype=torch.float32,
        )

    return data