from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

rdBase.DisableLog("rdApp.warning")
rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.info")
rdBase.DisableLog("rdApp.debug")


ATOM_TYPES = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


@dataclass
class MappedV2Featurizer:
    feature_mode: str = "reaction_mapped_v2"
    radius: int = 2
    n_bits: int = 2048
    reaction_col: str = "Reaction"
    mapped_col: str = "MappedReaction"
    largest_col: str = "largest_molecule"

    def __post_init__(self) -> None:
        valid_modes = {
            "mapped_v2",
            "reaction_mapped_v2",
            "reaction_mapped_largest_v2",
        }
        if self.feature_mode not in valid_modes:
            raise ValueError(
                f"Invalid feature_mode={self.feature_mode}. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        self.generator = GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)

        self.total_molecules_seen = 0
        self.failed_molecules = 0
        self.failed_reactions = 0
        self.failed_mapped_reactions = 0

    def fit(self, data: pd.DataFrame) -> "MappedV2Featurizer":
        _ = data
        return self

    def _empty_fp(self) -> np.ndarray:
        return np.zeros((self.n_bits,), dtype=np.float32)

    def _smiles_to_mol(self, smiles: str) -> Chem.Mol | None:
        self.total_molecules_seen += 1

        if not smiles or not isinstance(smiles, str):
            self.failed_molecules += 1
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self.failed_molecules += 1
            return None

        return mol

    def _mol_to_fp(self, mol: Chem.Mol, from_atoms: list[int] | None = None) -> np.ndarray:
        arr = self._empty_fp()

        if from_atoms is None:
            fp = self.generator.GetFingerprint(mol)
        else:
            if len(from_atoms) == 0:
                return arr
            fp = self.generator.GetFingerprint(mol, fromAtoms=from_atoms)

        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _smiles_to_fp(self, smiles: str) -> np.ndarray:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return self._empty_fp()
        return self._mol_to_fp(mol)

    def _multi_smiles_to_fp(self, smiles_block: str) -> np.ndarray:
        total = self._empty_fp()

        if not smiles_block or not isinstance(smiles_block, str):
            return total

        for part in [s for s in smiles_block.split(".") if s]:
            total += self._smiles_to_fp(part)

        return total

    def _reaction_fp(self, reaction: str) -> np.ndarray:
        zero = self._empty_fp()

        if not reaction or not isinstance(reaction, str) or ">>" not in reaction:
            self.failed_reactions += 1
            return zero

        try:
            reactants, products = reaction.split(">>", 1)
        except ValueError:
            self.failed_reactions += 1
            return zero

        reactant_fp = self._multi_smiles_to_fp(reactants)
        product_fp = self._multi_smiles_to_fp(products)

        return product_fp - reactant_fp

    def _atom_signature(self, atom: Chem.Atom) -> tuple:
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

    def _side_mols_and_info(
        self,
        smiles_block: str,
    ) -> tuple[list[Chem.Mol], dict[int, tuple], dict[tuple[int, int], Chem.BondType]]:
        mols: list[Chem.Mol] = []
        atom_info: dict[int, tuple] = {}
        bond_info: dict[tuple[int, int], Chem.BondType] = {}

        if not smiles_block or not isinstance(smiles_block, str):
            return mols, atom_info, bond_info

        for part in [s for s in smiles_block.split(".") if s]:
            mol = self._smiles_to_mol(part)
            if mol is None:
                continue

            mols.append(mol)

            for atom in mol.GetAtoms():
                atom_map = atom.GetAtomMapNum()
                if atom_map <= 0:
                    continue
                atom_info[atom_map] = self._atom_signature(atom)

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetAtomMapNum()
                a2 = bond.GetEndAtom().GetAtomMapNum()

                if a1 <= 0 or a2 <= 0:
                    continue

                key = tuple(sorted((a1, a2)))
                bond_info[key] = bond.GetBondType()

        return mols, atom_info, bond_info

    def _center_fp_from_block(self, smiles_block: str, changed_maps: set[int]) -> np.ndarray:
        total = self._empty_fp()

        if not smiles_block or not isinstance(smiles_block, str):
            return total

        for part in [s for s in smiles_block.split(".") if s]:
            mol = self._smiles_to_mol(part)
            if mol is None:
                continue

            center_atoms = [
                atom.GetIdx()
                for atom in mol.GetAtoms()
                if atom.GetAtomMapNum() in changed_maps
            ]

            total += self._mol_to_fp(mol, from_atoms=center_atoms)

        return total

    def _descriptor_vector(
        self,
        reactant_atom_info: dict[int, tuple],
        product_atom_info: dict[int, tuple],
        reactant_bond_info: dict[tuple[int, int], Chem.BondType],
        product_bond_info: dict[tuple[int, int], Chem.BondType],
        changed_maps: set[int],
    ) -> np.ndarray:
        features: list[float] = []

        # Basic reaction center counts
        all_bonds = set(reactant_bond_info) | set(product_bond_info)
        formed_bonds = [b for b in all_bonds if b not in reactant_bond_info and b in product_bond_info]
        broken_bonds = [b for b in all_bonds if b in reactant_bond_info and b not in product_bond_info]
        changed_bonds = [
            b
            for b in all_bonds
            if b in reactant_bond_info
            and b in product_bond_info
            and reactant_bond_info[b] != product_bond_info[b]
        ]

        features.extend(
            [
                float(len(changed_maps)),
                float(len(formed_bonds)),
                float(len(broken_bonds)),
                float(len(changed_bonds)),
            ]
        )

        # Changed atom type counts
        for atomic_num in ATOM_TYPES:
            count = 0
            for atom_map in changed_maps:
                sig = product_atom_info.get(atom_map) or reactant_atom_info.get(atom_map)
                if sig and sig[0] == atomic_num:
                    count += 1
            features.append(float(count))

        # Aromatic / charged / chiral changed atom counts
        aromatic_count = 0
        charged_count = 0
        chiral_count = 0

        for atom_map in changed_maps:
            sig = product_atom_info.get(atom_map) or reactant_atom_info.get(atom_map)
            if not sig:
                continue

            atomic_num, formal_charge, is_aromatic, chiral_tag, _ = sig

            if is_aromatic:
                aromatic_count += 1
            if formal_charge != 0:
                charged_count += 1
            if chiral_tag != "CHI_UNSPECIFIED":
                chiral_count += 1

        features.extend(
            [
                float(aromatic_count),
                float(charged_count),
                float(chiral_count),
            ]
        )

        # Bond type counts: formed / broken / changed
        for bond_type in BOND_TYPES:
            features.append(float(sum(product_bond_info[b] == bond_type for b in formed_bonds)))

        for bond_type in BOND_TYPES:
            features.append(float(sum(reactant_bond_info[b] == bond_type for b in broken_bonds)))

        for bond_type in BOND_TYPES:
            features.append(
                float(
                    sum(
                        reactant_bond_info[b] == bond_type or product_bond_info[b] == bond_type
                        for b in changed_bonds
                    )
                )
            )

        return np.asarray(features, dtype=np.float32)

    def _mapped_v2_fp(self, mapped_reaction: str) -> np.ndarray:
        zero_fp = self._empty_fp()

        if (
            not mapped_reaction
            or not isinstance(mapped_reaction, str)
            or ">>" not in mapped_reaction
        ):
            self.failed_mapped_reactions += 1
            return np.concatenate([zero_fp, zero_fp, np.zeros(29, dtype=np.float32)])

        try:
            reactants, products = mapped_reaction.split(">>", 1)
        except ValueError:
            self.failed_mapped_reactions += 1
            return np.concatenate([zero_fp, zero_fp, np.zeros(29, dtype=np.float32)])

        _, reactant_atom_info, reactant_bond_info = self._side_mols_and_info(reactants)
        _, product_atom_info, product_bond_info = self._side_mols_and_info(products)

        all_maps = set(reactant_atom_info) | set(product_atom_info)
        changed_maps = {
            atom_map
            for atom_map in all_maps
            if reactant_atom_info.get(atom_map) != product_atom_info.get(atom_map)
        }

        all_bonds = set(reactant_bond_info) | set(product_bond_info)
        for bond_key in all_bonds:
            if reactant_bond_info.get(bond_key) != product_bond_info.get(bond_key):
                changed_maps.update(bond_key)

        if not changed_maps:
            descriptor = np.zeros(29, dtype=np.float32)
            return np.concatenate([zero_fp, zero_fp, descriptor]).astype(np.float32)

        reactant_center = self._center_fp_from_block(reactants, changed_maps)
        product_center = self._center_fp_from_block(products, changed_maps)

        center_diff = product_center - reactant_center
        center_abs_diff = np.abs(center_diff)

        descriptor = self._descriptor_vector(
            reactant_atom_info=reactant_atom_info,
            product_atom_info=product_atom_info,
            reactant_bond_info=reactant_bond_info,
            product_bond_info=product_bond_info,
            changed_maps=changed_maps,
        )

        return np.concatenate([center_diff, center_abs_diff, descriptor]).astype(np.float32)

    def _largest_fp(self, smiles: str) -> np.ndarray:
        return self._smiles_to_fp(smiles)

    def _get_existing_col(self, data: pd.DataFrame, candidates: list[str]) -> str:
        for col in candidates:
            if col in data.columns:
                return col
        raise ValueError(f"Could not find any of these columns: {candidates}")

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("MappedV2Featurizer expects the full split dataframe.")

        reaction_col = self._get_existing_col(data, [self.reaction_col, "reaction"])
        mapped_col = self._get_existing_col(data, [self.mapped_col, "mapped_reaction"])

        reaction_x = np.vstack(
            [self._reaction_fp(x) for x in data[reaction_col].fillna("")]
        ).astype(np.float32)

        mapped_x = np.vstack(
            [self._mapped_v2_fp(x) for x in data[mapped_col].fillna("")]
        ).astype(np.float32)

        if self.feature_mode == "mapped_v2":
            return mapped_x.astype(np.float32)
        
        if self.feature_mode == "reaction_mapped_v2":
            return np.hstack([reaction_x, mapped_x]).astype(np.float32)

        largest_col = self._get_existing_col(
            data,
            [
                self.largest_col,
                "largest_molecule",
                "largest_molecul",
                "LargestMolecule",
            ],
        )

        largest_x = np.vstack(
            [self._largest_fp(x) for x in data[largest_col].fillna("")]
        ).astype(np.float32)

        return np.hstack([reaction_x, mapped_x, largest_x]).astype(np.float32)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def summary(self) -> dict[str, float | int | str]:
        molecule_failure_rate = (
            self.failed_molecules / self.total_molecules_seen
            if self.total_molecules_seen > 0
            else 0.0
        )

        return {
            "feature_mode": self.feature_mode,
            "radius": self.radius,
            "n_bits": self.n_bits,
            "failed_reactions": self.failed_reactions,
            "failed_mapped_reactions": self.failed_mapped_reactions,
            "total_molecules_seen": self.total_molecules_seen,
            "failed_molecules": self.failed_molecules,
            "molecule_failure_rate": molecule_failure_rate,
        }