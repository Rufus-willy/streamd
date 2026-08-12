"""Tests for fix_zero_coordinate_hydrogens in ligand_preparation."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from streamd.preparation.ligand_preparation import fix_zero_coordinate_hydrogens

# These tests require a real RDKit build with 3D embedding capabilities.
pytestmark = pytest.mark.skipif(
    not callable(getattr(AllChem, "EmbedMolecule", None)),
    reason="RDKit 3D embedding is not available",
)


def _count_zero_coordinate_atoms(mol):
    """Return how many atoms sit at (or extremely close to) the origin."""
    conf = mol.GetConformer()
    n = 0
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        if abs(pos.x) < 1e-6 and abs(pos.y) < 1e-6 and abs(pos.z) < 1e-6:
            n += 1
    return n


def _build_hexane_with_zeroed_hydrogens(num_zero=3):
    """Build a hexane molecule with valid 3D coords then zero some hydrogens."""
    mol = Chem.MolFromSmiles("CCCCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    zeroed = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            conf.SetAtomPosition(atom.GetIdx(), (0.0, 0.0, 0.0))
            zeroed.append(atom.GetIdx())
            if len(zeroed) >= num_zero:
                break
    return mol, zeroed


def test_fixes_zero_coordinate_hydrogens():
    """Hydrogens pinned to the origin are regenerated after the fix."""
    mol, zeroed = _build_hexane_with_zeroed_hydrogens(num_zero=3)
    assert zeroed  # we did introduce zero-coordinate atoms
    assert _count_zero_coordinate_atoms(mol) == len(zeroed)

    mol = fix_zero_coordinate_hydrogens(mol)

    assert _count_zero_coordinate_atoms(mol) == 0


def test_normal_molecule_unchanged():
    """A molecule without zero-coordinate atoms is returned untouched."""
    mol = Chem.MolFromSmiles("CCCCCC")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    assert _count_zero_coordinate_atoms(mol) == 0

    result = fix_zero_coordinate_hydrogens(mol)

    # No re-embedding should happen: the same object is returned unchanged.
    assert result is mol
    assert _count_zero_coordinate_atoms(result) == 0


def test_no_conformer_returns_mol():
    """A molecule lacking any conformer is returned as-is without error."""
    mol = Chem.MolFromSmiles("CCCCCC")
    mol = Chem.AddHs(mol)
    assert mol.GetNumConformers() == 0

    result = fix_zero_coordinate_hydrogens(mol)

    assert result is mol
