import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import openfermion
import openfermion.chem
import requests
from openfermion import (
    MolecularData,
    get_fermion_operator,
    jordan_wigner,
)
from openfermion.chem import periodic_hash_table
from openfermion.ops import QubitOperator
from openfermionpyscf import run_pyscf
from pubchempy import Compound


from .abstract import BenchmarkStateVector


def qubitop_term_to_pauli_string(term_ops: Tuple[Tuple[int, str], ...]) -> str:
    """
    OpenFermion QubitOperator term key is like ((q0,'X'), (q2,'Y'), ...)
    Return a canonical string "X0 Y2 ..." sorted by qubit index.
    """
    if term_ops == ():
        return "I"
    # sort by index
    term_ops = tuple(sorted(term_ops, key=lambda x: x[0]))
    return " ".join([f"{p}{i}" for i, p in term_ops])


def build_qubit_hamiltonian_jw(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
) -> QubitOperator:
    """
    Returns OpenFermion QubitOperator after Jordan-Wigner mapping.

    Args:
        geometry (List[Tuple[str, Tuple[float, float, float]]]): Molecular geometry.
        basis (str): Basis set for the quantum chemistry calculation. Default is "sto-3g".
        charge (int): Molecular charge. Default is 0.
        multiplicity (int): Spin multiplicity. Default is 1.

    Returns:
        QubitOperator: The qubit Hamiltonian after Jordan-Wigner transformation.
    """

    # Create molecular data object
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
    )

    # Run quantum chemistry calculations
    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
    )

    # Get second-quantized molecular Hamiltonian and convert to FermionOperator
    molecular_ham = molecule.get_molecular_hamiltonian()
    fermion_op = get_fermion_operator(molecular_ham)

    # Perform Jordan-Wigner transformation to get QubitOperator
    qubit_op = jordan_wigner(fermion_op)
    qubit_op.compress()  # Combine like terms and drop near-zero coefficients
    return qubit_op


# ---- LCU term extraction ----
@dataclass(frozen=True)
class LcuTerm:
    pauli: str  # e.g. "X0 Y2 Z3" or "I" for identity
    coeff: complex  # original coefficient (for reference)


def get_prep_state_from_qubit_operator(
    qubit_op: QubitOperator, exclude_identity_term: bool, drop_tol: float = 1e-12
) -> np.ndarray:

    terms: List[LcuTerm] = list()
    for term_ops, coeff in qubit_op.terms.items():
        if abs(coeff) < drop_tol:
            continue
        pauli = qubitop_term_to_pauli_string(term_ops)
        if exclude_identity_term and pauli == "I":
            continue
        terms.append(LcuTerm(pauli=pauli, coeff=coeff))

    # For LCU it’s often convenient to sort by descending alpha
    terms.sort(key=lambda t: abs(t.coeff), reverse=True)
    sv = np.zeros(2 ** math.ceil(np.log2(len(terms))), dtype=np.complex128)

    for i, term in enumerate(terms):
        sv[i] = term.coeff
    sv /= np.linalg.norm(sv)
    return sv


def infer_multiplicity_from_electron_count(
    geometry: List[Tuple[str, Tuple[float, float, float]]],
    charge: int = 0,
) -> int:
    """Fallback: infer minimal multiplicity from total electron count."""
    total_electrons = 0
    for atom, _ in geometry:
        total_electrons += periodic_hash_table[atom]
    total_electrons -= charge

    # Even electrons → singlet, odd → doublet (minimal spin assumption)
    if total_electrons % 2 == 0:
        return 1
    else:
        return 2


_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def _cas_from_pubchem_compound(compound: Compound) -> str | None:
    try:
        for syn in compound.synonyms or []:
            if _CAS_RE.match(syn):
                return syn
    except Exception:
        return None
    return None


def _cas_from_pubchem_name(name: str) -> str | None:
    try:
        compound = Compound.from_name(name)
        return _cas_from_pubchem_compound(compound)
    except Exception:
        return None


def _parse_multiplicity_from_html(html: str) -> int | None:
    """
    Parse CCCBDB 'Electronic Energy Levels' table and return degeneracy
    where Energy (cm^-1) == 0.
    """
    box_match = re.search(
        r'<div[^>]*class="box"[^>]*title="Electronic Energy Levels"[^>]*>(.*?)</div>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not box_match:
        return None

    box_html = box_match.group(1)
    table_match = re.search(
        r"<table\b.*?>.*?</table>", box_html, flags=re.DOTALL | re.IGNORECASE
    )
    if not table_match:
        return None

    table_html = table_match.group(0)
    rows = re.findall(r"<tr\b.*?>.*?</tr>", table_html, flags=re.DOTALL | re.IGNORECASE)
    if not rows:
        return None

    def _clean_cell(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    header_cells = re.findall(
        r"<t[dh]\b.*?>.*?</t[dh]>", rows[0], flags=re.DOTALL | re.IGNORECASE
    )
    if not header_cells:
        return None

    headers = [_clean_cell(c).lower() for c in header_cells]

    def _find_col(needle: str) -> int | None:
        for i, h in enumerate(headers):
            if needle in h:
                return i
        return None

    energy_idx = _find_col("energy")
    degeneracy_idx = _find_col("degeneracy")
    if energy_idx is None or degeneracy_idx is None:
        return None

    def _parse_float(s: str) -> float | None:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None

    for row in rows[1:]:
        cells = re.findall(
            r"<t[dh]\b.*?>.*?</t[dh]>", row, flags=re.DOTALL | re.IGNORECASE
        )
        if len(cells) <= max(energy_idx, degeneracy_idx):
            continue
        clean_cells = [_clean_cell(c) for c in cells]

        energy_val = _parse_float(clean_cells[energy_idx])
        if energy_val is None or abs(energy_val) > 1e-9:
            continue

        degeneracy_val = _parse_float(clean_cells[degeneracy_idx])
        if degeneracy_val is None:
            continue

        return int(round(degeneracy_val))

    return None


def fetch_multiplicity_from_cccbdb(name: str, cas: str | None = None) -> int | None:
    """
    Try to fetch ground-state multiplicity from NIST CCCBDB.
    Parses the term symbol like ^3Σg and extracts the leading integer (3).
    Returns None if not found.
    """
    search_url = "https://cccbdb.nist.gov/exp2.asp"
    params = {"casno": cas or name}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "close",
    }

    r = requests.get(
        search_url,
        params=params,
        headers=headers,
        timeout=(5, 20),
    )
    r.raise_for_status()

    ret = _parse_multiplicity_from_html(r.text)
    return ret


class LcuPrepStatesBenchmark(BenchmarkStateVector):
    """Benchmark for LCU state preparation methods."""

    def __init__(self, id: str, multiplicity: int | None = None, charge: int = 0):
        # id should be either name of molecule or CID from PubChem
        cas = None
        if id.isdigit():
            # Fetch the molecule name from PubChem using the CID
            compound = Compound.from_cid(id)
            self.compound_name = compound.synonyms[0]
            cas = _cas_from_pubchem_compound(compound)
        else:
            self.compound_name = id
            cas = _cas_from_pubchem_name(self.compound_name)

        geometry = openfermion.chem.geometry_from_pubchem(self.compound_name)
        # Determine multiplicity
        if multiplicity is not None:
            final_multiplicity = multiplicity
        else:
            # Try CCCBDB first
            final_multiplicity = fetch_multiplicity_from_cccbdb(
                self.compound_name, cas=cas
            )

        self.qubit_op = build_qubit_hamiltonian_jw(
            geometry,
            charge=charge,
            multiplicity=final_multiplicity,
        )
        self.prep_sv = get_prep_state_from_qubit_operator(self.qubit_op, True)

    def __call__(self):
        return self.prep_sv
