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


# ============================================================
# Edge-v4 configuration
# ============================================================
# v4 node dim:
#   edge_v3 node dim = 39
#   replace shell features [is_1hop, is_2hop] with
#   [is_center_distance, is_1hop, is_2hop, is_3hop, is_far]
#   39 - 2 + 5 = 42
EDGE_V4_NODE_DIM = 42

# v4 edge dim:
#   edge_v2 base = 14
#   + changed_mapping_edge
#   + 1hop_mapping_edge
#   + 2hop_mapping_edge
#   + informative_far_mapping_edge
#   + context_edge
#   = 19
EDGE_V4_EDGE_DIM = 19


# Informative far rule.
# First version: keep chemically informative far unchanged mapping edges.
# If INFORMATIVE_FAR_TOP_K is None, keep all informative far edges.
# If it is an integer, keep at most top-k informative far edges by score.
INFORMATIVE_FAR_KEEP_HETERO = True
INFORMATIVE_FAR_KEEP_RING = True
INFORMATIVE_FAR_KEEP_AROMATIC = True
INFORMATIVE_FAR_KEEP_CHARGED = True
INFORMATIVE_FAR_TOP_K: int | None = None


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
                "atomic_num": float(atom.GetAtomicNum()),
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


def get_distance_shell_maps(
    changed_maps: set[int],
    adjacency: dict[int, set[int]],
) -> tuple[set[int], set[int], set[int], set[int]]:
    """
    Center-shell positional encoding for edge_v4:
        1-hop: direct neighbors of reaction-center atoms
        2-hop: neighbors of 1-hop atoms
        3-hop: neighbors of 2-hop atoms
        far: all mapped atoms outside center/1-hop/2-hop/3-hop

    Center atoms themselves are excluded from shell sets.
    """
    all_maps = set(adjacency.keys())

    one_hop = set()
    for amap in changed_maps:
        one_hop.update(adjacency.get(amap, set()))
    one_hop -= changed_maps

    two_hop = set()
    for amap in one_hop:
        two_hop.update(adjacency.get(amap, set()))
    two_hop -= changed_maps
    two_hop -= one_hop

    three_hop = set()
    for amap in two_hop:
        three_hop.update(adjacency.get(amap, set()))
    three_hop -= changed_maps
    three_hop -= one_hop
    three_hop -= two_hop

    far_maps = all_maps - changed_maps - one_hop - two_hop - three_hop

    return one_hop, two_hop, three_hop, far_maps


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


def atom_features_edge_v4(
    atom: Chem.Atom,
    side_flag: int,
    is_center: bool,
    is_1hop_from_center: bool,
    is_2hop_from_center: bool,
    is_3hop_from_center: bool,
    is_far_from_center: bool,
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
    formed_bonds: set[tuple[int, int]],
    broken_bonds: set[tuple[int, int]],
    changed_bonds: set[tuple[int, int]],
) -> list[float]:
    """
    Node feature v4.

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

    Added center-distance positional encoding:
        is_center_distance
        is_1hop_from_center
        is_2hop_from_center
        is_3hop_from_center
        is_far_from_center
        subtotal: 5

    Total:
        17 + 11 + 9 + 5 = 42 dims
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

    distance_features = [
        float(is_center),
        float(is_1hop_from_center),
        float(is_2hop_from_center),
        float(is_3hop_from_center),
        float(is_far_from_center),
    ]

    features = [
        *base_features,
        *basic_state_features,
        *node_change_features,
        *distance_features,
    ]

    if len(features) != EDGE_V4_NODE_DIM:
        raise ValueError(
            f"edge_v4 node feature dim mismatch: "
            f"got {len(features)}, expected {EDGE_V4_NODE_DIM}"
        )

    return features


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


def bond_features_edge_v4(
    bond_order_before: float = 0.0,
    bond_order_after: float = 0.0,
    is_reactant_edge: bool = False,
    is_product_edge: bool = False,
    is_reaction_edge: bool = False,
    is_center_edge: bool = False,
    is_formed_bond: bool = False,
    is_broken_bond: bool = False,
    is_order_changed: bool = False,
    is_changed_mapping_edge: bool = False,
    is_1hop_mapping_edge: bool = False,
    is_2hop_mapping_edge: bool = False,
    is_informative_far_mapping_edge: bool = False,
    is_context_edge: bool = False,
) -> list[float]:
    """
    Edge feature v4.

    Original edge_v2:
        14 dims

    Added typed mapping/context-edge indicators:
        is_changed_mapping_edge
        is_1hop_mapping_edge
        is_2hop_mapping_edge
        is_informative_far_mapping_edge
        is_context_edge

    Total:
        14 + 5 = 19 dims

    Important:
        No manually fixed continuous edge weight is used.
        Edge importance is learned by GATConv from edge_attr.
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

    features = [
        *base,
        float(is_changed_mapping_edge),
        float(is_1hop_mapping_edge),
        float(is_2hop_mapping_edge),
        float(is_informative_far_mapping_edge),
        float(is_context_edge),
    ]

    if len(features) != EDGE_V4_EDGE_DIM:
        raise ValueError(
            f"edge_v4 edge feature dim mismatch: "
            f"got {len(features)}, expected {EDGE_V4_EDGE_DIM}"
        )

    return features


def merged_atom_context(
    amap: int,
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Merge reactant/product atom context for an atom map.

    Product context is allowed to fill missing fields. For most unchanged far
    atoms, reactant/product contexts are similar.
    """
    context = {}
    context.update(product_context.get(amap, {}))
    context.update(reactant_context.get(amap, {}))
    return context


def informative_far_score(
    amap: int,
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
) -> float:
    """
    Score an unchanged far mapped atom.

    This is only used when INFORMATIVE_FAR_TOP_K is not None.
    The rule keeps chemically informative far atoms:
        hetero atoms
        ring atoms
        aromatic atoms
        charged atoms
    """
    context = merged_atom_context(
        amap=amap,
        reactant_context=reactant_context,
        product_context=product_context,
    )

    atomic_num = int(context.get("atomic_num", 0.0))
    formal_charge = float(context.get("formal_charge", 0.0))
    is_aromatic = float(context.get("is_aromatic", 0.0))
    is_in_ring = float(context.get("is_in_ring", 0.0))

    score = 0.0

    # N, O, S, P, F, Cl, Br, I and other non-C/H atoms.
    if INFORMATIVE_FAR_KEEP_HETERO and atomic_num not in {0, 1, 6}:
        score += 3.0

    if INFORMATIVE_FAR_KEEP_CHARGED and formal_charge != 0.0:
        score += 2.5

    if INFORMATIVE_FAR_KEEP_AROMATIC and is_aromatic > 0.0:
        score += 2.0

    if INFORMATIVE_FAR_KEEP_RING and is_in_ring > 0.0:
        score += 1.5

    return score


def is_informative_far_atom(
    amap: int,
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
) -> bool:
    return informative_far_score(
        amap=amap,
        reactant_context=reactant_context,
        product_context=product_context,
    ) > 0.0


def select_informative_far_maps(
    far_candidate_maps: set[int],
    reactant_context: dict[int, dict[str, float]],
    product_context: dict[int, dict[str, float]],
) -> set[int]:
    scored = []

    for amap in far_candidate_maps:
        score = informative_far_score(
            amap=amap,
            reactant_context=reactant_context,
            product_context=product_context,
        )

        if score > 0.0:
            scored.append((amap, score))

    if INFORMATIVE_FAR_TOP_K is None:
        return {amap for amap, _ in scored}

    scored = sorted(scored, key=lambda x: (-x[1], x[0]))
    return {amap for amap, _ in scored[:INFORMATIVE_FAR_TOP_K]}


def add_edge(
    edge_index: list[list[int]],
    edge_attr: list[list[float]],
    i: int,
    j: int,
    features: list[float],
) -> None:
    edge_index.append([i, j])
    edge_attr.append(features)


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

    one_hop_maps, two_hop_maps, three_hop_maps, far_maps = (
        get_distance_shell_maps(
            changed_maps=changed_maps,
            adjacency=merged_adj,
        )
    )

    x = []
    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
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
                is_3hop = amap in three_hop_maps if amap > 0 else False
                is_far = amap in far_maps if amap > 0 else False

                x.append(
                    atom_features_edge_v4(
                        atom=atom,
                        side_flag=side_flag,
                        is_center=is_center,
                        is_1hop_from_center=is_1hop,
                        is_2hop_from_center=is_2hop,
                        is_3hop_from_center=is_3hop,
                        is_far_from_center=is_far,
                        reactant_context=reactant_context,
                        product_context=product_context,
                        formed_bonds=formed_bonds,
                        broken_bonds=broken_bonds,
                        changed_bonds=changed_bonds,
                    )
                )

                center_mask.append(float(is_center))
                center_local_mask.append(
                    float(is_center or is_1hop or is_2hop or is_3hop)
                )

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

                bf = bond_features_edge_v4(
                    bond_order_before=bond_order_before,
                    bond_order_after=bond_order_after,
                    is_reactant_edge=is_reactant_side,
                    is_product_edge=is_product_side,
                    is_reaction_edge=False,
                    is_center_edge=is_center_edge,
                    is_formed_bond=is_formed_bond,
                    is_broken_bond=is_broken_bond,
                    is_order_changed=is_order_changed,
                    is_changed_mapping_edge=False,
                    is_1hop_mapping_edge=False,
                    is_2hop_mapping_edge=False,
                    is_informative_far_mapping_edge=False,
                    is_context_edge=False,
                )

                add_edge(edge_index, edge_attr, i, j, bf)
                add_edge(edge_index, edge_attr, j, i, bf)

    shared_maps = (
        set(reactant_map_to_node.keys())
        & set(product_map_to_node.keys())
    )

    # Far candidates are shared mapped atoms that are not center/1-hop/2-hop.
    # We allow 3-hop/far atoms to become informative far if chemically meaningful.
    far_candidate_maps = {
        amap
        for amap in shared_maps
        if (
            amap not in changed_maps
            and amap not in one_hop_maps
            and amap not in two_hop_maps
        )
    }

    informative_far_maps = select_informative_far_maps(
        far_candidate_maps=far_candidate_maps,
        reactant_context=reactant_context,
        product_context=product_context,
    )

    num_changed_mapping_edges = 0
    num_1hop_mapping_edges = 0
    num_2hop_mapping_edges = 0
    num_far_candidates = len(far_candidate_maps)
    num_far_kept = 0
    num_far_dropped = 0

    for amap in shared_maps:
        i = reactant_map_to_node[amap]
        j = product_map_to_node[amap]

        is_center = amap in changed_maps
        is_1hop = amap in one_hop_maps
        is_2hop = amap in two_hop_maps

        is_changed_mapping_edge = False
        is_1hop_mapping_edge = False
        is_2hop_mapping_edge = False
        is_informative_far_mapping_edge = False

        keep_reaction_edge = False

        if is_center:
            keep_reaction_edge = True
            is_changed_mapping_edge = True
            num_changed_mapping_edges += 1

        elif is_1hop:
            keep_reaction_edge = True
            is_1hop_mapping_edge = True
            num_1hop_mapping_edges += 1

        elif is_2hop:
            keep_reaction_edge = True
            is_2hop_mapping_edge = True
            num_2hop_mapping_edges += 1

        elif amap in informative_far_maps:
            keep_reaction_edge = True
            is_informative_far_mapping_edge = True
            num_far_kept += 1

        else:
            if amap in far_candidate_maps:
                num_far_dropped += 1

        if not keep_reaction_edge:
            continue

        rf = bond_features_edge_v4(
            bond_order_before=0.0,
            bond_order_after=0.0,
            is_reactant_edge=False,
            is_product_edge=False,
            is_reaction_edge=True,
            is_center_edge=is_center,
            is_formed_bond=False,
            is_broken_bond=False,
            is_order_changed=False,
            is_changed_mapping_edge=is_changed_mapping_edge,
            is_1hop_mapping_edge=is_1hop_mapping_edge,
            is_2hop_mapping_edge=is_2hop_mapping_edge,
            is_informative_far_mapping_edge=is_informative_far_mapping_edge,
            is_context_edge=False,
        )

        add_edge(edge_index, edge_attr, i, j, rf)
        add_edge(edge_index, edge_attr, j, i, rf)

    if len(x) == 0:
        raise ValueError("Empty reaction graph.")

    if len(edge_index) == 0:
        # Rare single-atom or disconnected case. Add a zero self-loop so that
        # downstream GATConv always receives a valid edge_attr matrix.
        edge_index.append([0, 0])
        edge_attr.append(
            bond_features_edge_v4(
                is_reactant_edge=False,
                is_product_edge=False,
                is_reaction_edge=False,
                is_center_edge=False,
                is_formed_bond=False,
                is_broken_bond=False,
                is_order_changed=False,
                is_changed_mapping_edge=False,
                is_1hop_mapping_edge=False,
                is_2hop_mapping_edge=False,
                is_informative_far_mapping_edge=False,
                is_context_edge=False,
            )
        )

    x_arr = np.asarray(x, dtype=np.float32)
    edge_attr_arr = np.asarray(edge_attr, dtype=np.float32)

    if x_arr.shape[1] != EDGE_V4_NODE_DIM:
        raise ValueError(
            f"edge_v4 node_dim mismatch: got {x_arr.shape[1]}, "
            f"expected {EDGE_V4_NODE_DIM}"
        )

    if edge_attr_arr.shape[1] != EDGE_V4_EDGE_DIM:
        raise ValueError(
            f"edge_v4 edge_dim mismatch: got {edge_attr_arr.shape[1]}, "
            f"expected {EDGE_V4_EDGE_DIM}"
        )

    far_keep_ratio = (
        float(num_far_kept) / float(num_far_candidates)
        if num_far_candidates > 0
        else 0.0
    )

    data = Data(
        x=torch.tensor(x_arr, dtype=torch.float32),
        edge_index=torch.tensor(
            edge_index,
            dtype=torch.long,
        ).t().contiguous(),
        edge_attr=torch.tensor(edge_attr_arr, dtype=torch.float32),
    )

    data.center_mask = torch.tensor(
        center_mask,
        dtype=torch.float32,
    )

    data.center_local_mask = torch.tensor(
        center_local_mask,
        dtype=torch.float32,
    )

    # Statistics for later visualization / error analysis.
    data.num_shared_maps = torch.tensor(
        [float(len(shared_maps))],
        dtype=torch.float32,
    )
    data.num_changed_mapping_edges = torch.tensor(
        [float(num_changed_mapping_edges)],
        dtype=torch.float32,
    )
    data.num_1hop_mapping_edges = torch.tensor(
        [float(num_1hop_mapping_edges)],
        dtype=torch.float32,
    )
    data.num_2hop_mapping_edges = torch.tensor(
        [float(num_2hop_mapping_edges)],
        dtype=torch.float32,
    )
    data.num_far_candidates = torch.tensor(
        [float(num_far_candidates)],
        dtype=torch.float32,
    )
    data.num_far_kept = torch.tensor(
        [float(num_far_kept)],
        dtype=torch.float32,
    )
    data.num_far_dropped = torch.tensor(
        [float(num_far_dropped)],
        dtype=torch.float32,
    )
    data.far_keep_ratio = torch.tensor(
        [float(far_keep_ratio)],
        dtype=torch.float32,
    )

    # Useful for documenting the top-k / rule setting used in this graph.
    data.informative_far_top_k = torch.tensor(
        [-1.0 if INFORMATIVE_FAR_TOP_K is None else float(INFORMATIVE_FAR_TOP_K)],
        dtype=torch.float32,
    )

    if y is not None:
        data.y = torch.tensor(
            [int(y)],
            dtype=torch.float32,
        )

    return data