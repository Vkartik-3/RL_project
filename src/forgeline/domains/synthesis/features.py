"""128-d state vector: 8 normalised chemistry features + 100-bit Morgan fingerprint + padding."""

from __future__ import annotations

from typing import Dict, List

import torch

from forgeline.domains.synthesis.reward import extract_outcomes

MOLECULE_SMILES = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Naproxen": "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ketoprofen": "CC(C(=O)O)c1ccc(cc1)C(=O)c1ccccc1",
}
_FP_CACHE: Dict[str, List[float]] = {}
STATE_DIM = 128


def molecule_fingerprint(name: str, n_bits: int = 100) -> List[float]:
    """Morgan fingerprint (radius 2) via RDKit when available; zeros otherwise."""
    if name in _FP_CACHE:
        return _FP_CACHE[name]
    smiles = MOLECULE_SMILES.get(name)
    bits = [0.0] * n_bits
    if smiles is not None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
                bits = [float(b) for b in fp.ToBitString()]
        except Exception:
            bits = [0.0] * n_bits
    _FP_CACHE[name] = bits
    return bits


def encode_trajectory(record: Dict) -> torch.Tensor:
    o = extract_outcomes(record)
    features = [o["temperature"] / 200.0, o["time"] / 12.0, min(o["catalyst"], 1.0), o["solvent"] / 6.0,
                o["yield"], o["selectivity"], 1.0 - o["safety_risk"], o["steps"] / 10.0]
    state = torch.zeros(STATE_DIM, dtype=torch.float32)
    state[:8] = torch.tensor(features, dtype=torch.float32)
    state[8:108] = torch.tensor(molecule_fingerprint(record.get("molecule", "")), dtype=torch.float32)
    return state
