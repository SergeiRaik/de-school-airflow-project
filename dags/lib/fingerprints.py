#  - 'ECFP4' - function rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
#  - 'ECFP6' - function rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
#  - 'MACCS' - function rdkit.Chem.MACCSkeys.GenMACCSKeys(mol)

from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

ECFP4_GENERATOR = GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

ECFP6_GENERATOR = GetMorganGenerator(
    radius=3,
    fpSize=2048,
)


def ecfp4(mol):
    return ECFP4_GENERATOR.GetFingerprint(mol)


def ecfp6(mol):
    return ECFP6_GENERATOR.GetFingerprint(mol)


FINGERPRINTS = {
    "ECFP4": ecfp4,
    "ECFP6": ecfp6,
    "MACCS": MACCSkeys.GenMACCSKeys,
}

def get_fingerprint_function(name):
    try:
        return FINGERPRINTS[name]
    except KeyError:
        raise ValueError(f"Fingerprint {name} is not registered.")
