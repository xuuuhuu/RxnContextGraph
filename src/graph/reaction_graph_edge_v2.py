from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_LIST = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]


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


def atom_features(
    atom: Chem.Atom,
    side_flag: int,
    is_center: bool = False,
) -> list[float]:
    symbol = atom.GetSymbol()

    atom_onehot = [1.0 if symbol == a else 0.0 for a in ATOM_LIST]

    other = 0.0 if symbol in ATOM_LIST else 1.0

    return atom_onehot + [
        other,
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetDegree()) / 6.0,
        float(atom.GetFormalCharge()),
        float(atom.GetIsAromatic()),
        float(side_flag),      # 0 = reactant, 1 = product
        float(is_center),      # reaction-center atom flag
    ]


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
        float(order == 1.0),  # single
        float(order == 2.0),  # double
        float(order == 3.0),  # triple
        float(order == 1.5),  # aromatic
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
    Edge feature v2.

    Feature order:
    0-3   : bond type one-hot
            [single, double, triple, aromatic]
    4     : is_reactant_edge
    5     : is_product_edge
    6     : is_reaction_edge
    7     : is_center_edge
    8     : is_formed_bond
    9     : is_broken_bond
    10    : is_order_changed
    11    : bond_order_before
    12    : bond_order_after
    13    : bond_order_delta
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


def split_side(side_smiles: str) -> list[str]:
    if not isinstance(side_smiles, str) or not side_smiles:
        return []

    return [s for s in side_smiles.split(".") if s]


def collect_mapped_info(
    smiles_block: str,
) -> tuple[
    dict[int, tuple],
    dict[tuple[int, int], Chem.BondType],
]:
    atom_info: dict[int, tuple] = {}
    bond_info: dict[tuple[int, int], Chem.BondType] = {}

    for smi in split_side(smiles_block):
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue

        for atom in mol.GetAtoms():
            amap = atom.GetAtomMapNum()

            if amap > 0:
                atom_info[amap] = atom_signature(atom)

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetAtomMapNum()
            a2 = bond.GetEndAtom().GetAtomMapNum()

            if a1 > 0 and a2 > 0:
                key = tuple(sorted((a1, a2)))
                bond_info[key] = bond.GetBondType()

    return atom_info, bond_info


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
        if reactant_atom_info.get(amap)
        != product_atom_info.get(amap)
    }

    all_bonds = set(reactant_bond_info) | set(product_bond_info)

    formed_bonds = {
        b
        for b in all_bonds
        if b not in reactant_bond_info
        and b in product_bond_info
    }

    broken_bonds = {
        b
        for b in all_bonds
        if b in reactant_bond_info
        and b not in product_bond_info
    }

    changed_bonds = {
        b
        for b in all_bonds
        if (
            b in reactant_bond_info
            and b in product_bond_info
            and reactant_bond_info[b]
            != product_bond_info[b]
        )
    }

    reaction_center_bonds = (
        formed_bonds
        | broken_bonds
        | changed_bonds
    )

    for b in reaction_center_bonds:
        changed_maps.update(b)

    return (
        changed_maps,
        reaction_center_bonds,
        formed_bonds,
        broken_bonds,
        changed_bonds,
    )


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

    reactant_atom_info, reactant_bond_info = collect_mapped_info(
        reactant_block
    )

    product_atom_info, product_bond_info = collect_mapped_info(
        product_block
    )

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

    x = []
    edge_index = []
    edge_attr = []
    center_mask = []

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

                is_center = (
                    amap in changed_maps
                    if amap > 0
                    else False
                )

                x.append(
                    atom_features(
                        atom,
                        side_flag,
                        is_center=is_center,
                    )
                )

                center_mask.append(float(is_center))

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

                # For reactant-side edges, the real visible bond is the before bond.
                # For product-side edges, the real visible bond is the after bond.
                bf = bond_features_edge_v2(
                    bond_order_before=bond_order_before,
                    bond_order_after=bond_order_after,
                    is_reactant_edge=is_reactant_side,
                    is_product_edge=is_product_side,
                    is_reaction_edge=False,
                    is_center_edge=is_center_edge,
                    is_formed_bond=is_formed_bond,
                    is_broken_bond=is_broken_bond,
                    is_order_changed=is_order_changed,
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

        rf = bond_features_edge_v2(
            bond_order_before=0.0,
            bond_order_after=0.0,
            is_reactant_edge=False,
            is_product_edge=False,
            is_reaction_edge=True,
            is_center_edge=is_center,
            is_formed_bond=False,
            is_broken_bond=False,
            is_order_changed=False,
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

    if y is not None:
        data.y = torch.tensor(
            [int(y)],
            dtype=torch.float32,
        )

    return data