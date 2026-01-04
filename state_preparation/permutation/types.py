from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd, log2
from typing import Iterable, List, Optional, Sequence, Tuple, Type

import cirq
import numpy as np
from cirq import Circuit

from state_preparation.gates.mcx.types import CanonMCXGate, MCXGateBase


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b) if a and b else 0


def _validate_mapping(mapping: Sequence[int]) -> None:
    n = len(mapping)
    if sorted(mapping) != list(range(0, n)):
        raise ValueError(
            f"`mapping` must be a permutation of 0..n-1 (0-based) with n={n} "
            f"Got: {mapping}"
        )


@dataclass(frozen=True)
class Permutation:
    """A permutation of {1, 2, ..., n}.

    Internally represented by a 1-based mapping `p` such that
    `p[i-1] = σ(i)`.

    Examples
    --------
    >>> # σ = (1 3 2)(5 6) on {1..6}
    >>> sigma = Permutation.from_cycles(6, [(1, 3, 2), (5, 6)])
    >>> sigma(1), sigma(2), sigma(3), sigma(5), sigma(6)
    (3, 1, 2, 6, 5)
    >>> sigma.cycles()
    [(1, 3, 2), (5, 6)]
    >>> # Composition: (self ∘ other)(i) = self(other(i))
    >>> tau = Permutation.from_cycles(6, [(1, 2, 3)])
    >>> (sigma @ tau).cycles()  # doctest: +NORMALIZE_WHITESPACE
    [(1, 2, 3), (5, 6)]  # result depends on ordering but cycles are correct
    >>> sigma.inverse().cycles()
    [(1, 2, 3), (5, 6)]
    >>> sigma.order()
    6
    """

    p: Tuple[int, ...]

    # -----------------------
    # Construction utilities
    # -----------------------
    def __init__(self, mapping: Sequence[int]):
        _validate_mapping(mapping)
        object.__setattr__(self, "p", tuple(mapping))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return NotImplementedError()
        return self.p == other.p

    @staticmethod
    def identity(n: int) -> "Permutation":
        return Permutation(tuple(range(0, n)))

    @staticmethod
    def from_cycles(n: int, cycles: Iterable[Iterable[int]]) -> "Permutation":
        """Construct from disjoint cycles in 0-based notation.

        Parameters
        ----------
        n : int
            The size of the underlying set {0, ..., n-1}.
        cycles : iterable of iterables
            Each inner iterable describes a cycle like (a1, a2, ..., ak).
            Fixed points may be omitted.
        """
        mapping = list(range(0, n))
        for cyc in cycles:
            cyc = list(cyc)
            if not cyc:
                continue
            for i, a in enumerate(cyc):
                b = cyc[(i + 1) % len(cyc)]
                if not (0 <= a <= n - 1 and 0 <= b <= n - 1):
                    raise ValueError(f"Cycle entries must be in 0..{n-1}")
                mapping[a] = b
        return Permutation(mapping)

    # -------------
    # Basic stats
    # -------------
    @property
    def n(self) -> int:
        return len(self.p)

    def __call__(self, i: int) -> int:
        if not (0 <= i <= self.n - 1):
            raise IndexError(f"Element {i} out of range 0..{self.n-1}")
        return self.p[i]

    def image(self, i: int) -> int:
        return self(i)

    # ------------------
    # Group operations
    # ------------------
    def compose(self, other: "Permutation") -> "Permutation":
        """Return self ∘ other (apply `other` first, then `self`)."""
        if self.n != other.n:
            raise ValueError("Permutations must act on the same set")
        return Permutation([self(other(i)) for i in range(0, self.n)])

    __matmul__ = compose  # use `@` for composition

    def inverse(self) -> "Permutation":
        inv = [0] * self.n
        for i, img in enumerate(self.p, start=0):
            inv[img] = i
        return Permutation(inv)

    def _cycles_in_tuple(
        self, remove_fixed: bool = True
    ) -> Tuple[Tuple[int, ...], ...]:
        seen = [False] * (self.n + 1)
        out: List[Tuple[int, ...]] = []
        for start in range(0, self.n):
            if seen[start]:
                continue
            cur = start
            cyc: List[int] = []
            while not seen[cur]:
                seen[cur] = True
                cyc.append(cur)
                cur = self.p[cur]
            if len(cyc) > 1 or not remove_fixed:
                out.append(cyc)
        return out

    # ---------------------------
    # Disjoint-cycle decomposition
    # ---------------------------
    def cycles(self, remove_fixed: bool = True) -> List[Cycle]:
        """Return the disjoint-cycle decomposition in 0-based notation.

        Parameters
        ----------
        remove_fixed : bool
            If True (default), omit 1-cycles (fixed points).
        """
        cycles = self._cycles_in_tuple(remove_fixed=remove_fixed)
        return [Cycle.from_cycle(self.n, cyc) for cyc in cycles]

    def sign(self) -> int:
        """Return +1 for even permutations, -1 for odd permutations."""
        # Sign = (-1)^(n - number_of_cycles) using full cycle decomposition
        num_cycles = len(self.cycles(remove_fixed=False))
        return 1 if ((self.n - num_cycles) % 2 == 0) else -1

    def order(self) -> int:
        """Return the order of the permutation (lcm of cycle lengths)."""
        lcm = 1
        for cyc in self.cycles(remove_fixed=True):
            assert isinstance(cyc, Cycle)
            lcm = _lcm(lcm, len(cyc))
        return max(lcm, 1)  # identity → 1

    def __str__(self) -> str:
        cyc = self.cycles(remove_fixed=True)
        if not cyc:
            return "()"  # identity
        return "".join(str(c) for c in cyc)

    __repr__ = __str__

    def decompose_into_two_disjoint_transpositions(
        self,
    ) -> Tuple[TranspositionsList, TranspositionsList]:
        set_one, set_two = set(), set()
        for cyc in self.cycles(remove_fixed=True):
            rho_prime, rho_double_prime = (
                cyc.decompose_into_two_disjoint_transpositions()
            )
            for t in rho_prime:
                set_one.add(t)
            for t in rho_double_prime:
                set_two.add(t)

        result_one, result_two = TranspositionsList(list(set_one)), TranspositionsList(
            list(set_two)
        )
        return result_one, result_two

    def size(self) -> int:
        """Return the size of the permutation, defined as the number of non-fixed points."""
        return sum(1 for i in range(self.n) if self(i) != i)

    def decompose_into_chunked_disjoint_transpositions(
        self, chunk_size: int
    ) -> List[DisjointTranspositions]:
        """Decompose the permutation into disjoint transpositions grouped by chunk size."""
        if not (chunk_size > 0 and (chunk_size & (chunk_size - 1)) == 0):
            raise ValueError("chunk_size must be a power of 2")

        chunked_transpositions = []
        result_one, result_two = self.decompose_into_two_disjoint_transpositions()

        for transpositions_list in [result_one, result_two]:
            chunks = [
                transpositions_list.transpositions[i : i + chunk_size]
                for i in range(0, len(transpositions_list.transpositions), chunk_size)
            ]
            for chunk in chunks:
                chunked_transpositions.append(DisjointTranspositions(chunk))

        return chunked_transpositions

    def index_extraction_based_decomposition_qc(
        self,
        qubits: List[cirq.LineQubit],
        mcx_gate_type: Type[MCXGateBase],
        do_validation: bool = False,
    ) -> Circuit:
        """Algorithm based in https://arxiv.org/abs/2406.16142"""

        assert 2 ** len(qubits) >= (self.n)
        qc = cirq.Circuit()
        chunk_size = max(int(2 ** np.floor(np.log2(np.log2(len(qubits)) / 4))), 1)
        for chunk in self.decompose_into_chunked_disjoint_transpositions(chunk_size):
            seq_transposes = SequentialTranspositions.from_num_transposes(
                chunk.n, chunk.m
            )

            index_extraction_map = chunk.index_extraction_map_qc(
                qubits, mcx_gate_type=mcx_gate_type
            )
            qc += index_extraction_map
            qc += seq_transposes.to_quantum_circuit(qubits, mcx_gate_type=mcx_gate_type)
            qc += index_extraction_map**-1

        # Validation
        if do_validation:
            for x in range(self.n):
                sv = cirq.one_hot(index=x, shape=2 ** len(qubits), dtype=np.complex128)
                qc_res = qc.final_state_vector(initial_state=sv, qubit_order=qubits)
                nonzero = np.where(qc_res > 1e-8)[0]

                assert len(nonzero) == 1
                assert nonzero[0] == self(x)

        return qc


def transposition(n: int, a: int, b: int) -> Optional[Transposition]:
    """Return the transposition (a b) in S_n.

    Disjoint transpositions are just disjoint 2-cycles in `Permutation.from_cycles`.
    """
    if a == b:
        return None
    return Transposition(n, a, b)


class Cycle(Permutation):

    def __init__(self, mapping: Sequence[int]):
        _validate_mapping(mapping)
        object.__setattr__(self, "p", tuple(mapping))
        assert self.is_identity() or len(self._cycles_in_tuple()) == 1

    def __len__(self) -> int:
        return sum(1 for i in self.p if self(i) != i)

    @property
    def cycle_repr(self) -> Tuple[int, ...]:
        if self.is_identity():
            return ()
        cycles = self._cycles_in_tuple()
        assert len(cycles) == 1
        return cycles[0]

    def is_identity(self) -> str:
        return self.p == Permutation.identity(self.n).p

    @staticmethod
    def from_cycle(n: int, cycle: Iterable[int]) -> Cycle:
        assert cycle
        mapping = list(range(0, n))
        cyc = list(cycle)
        for i, a in enumerate(cyc):
            b = cyc[(i + 1) % len(cyc)]
            if not (0 <= a <= n - 1 and 0 <= b <= n - 1):
                raise ValueError("Cycle entries must be in 0..n-1")
            mapping[a] = b
        return Cycle(mapping)

    def decompose_into_two_disjoint_transpositions(
        self,
    ) -> Tuple[TranspositionsList, TranspositionsList]:
        # inside Cycle.decompose_into_two_disjoint_transpositions(self)
        # returns [rho_prime, rho_double_prime] where each is a list of transposition Permutations
        if self.is_identity():
            raise ValueError("Identity cycle cannot be decomposed.")

        # Since Cycle guarantees a single nontrivial cycle, extract its sequence
        seqs = self._cycles_in_tuple()

        if not seqs:
            raise ValueError("No nontrivial cycle found.")

        seq = list(seqs[0])  # [x0, x1, ..., x_{m-1}]
        m = len(seq)
        n = self.n

        rho_prime: List[Permutation] = []
        rho_double_prime: List[Permutation] = []

        if m % 2 == 0:
            # even length: m = 2k
            k = m // 2
            # ρ' = (x0, x_{2k-1})(x1, x_{2k-2})...(x_{k-1}, x_k)
            for i in range(k):
                rho_prime.append(transposition(n, seq[i], seq[m - 1 - i]))
            # ρ'' = (x1, x_{2k-1})(x2, x_{2k-2})...(x_{k-1}, x_{k+1})
            for i in range(1, k):
                rho_double_prime.append(transposition(n, seq[i], seq[m - i]))
        else:
            # odd length: m = 2k + 1
            k = m // 2
            # ρ' = (x0, x_{2k})(x1, x_{2k-1})...(x_k, x_{k+1})
            for i in range(k + 1):
                rho_prime.append(transposition(n, seq[i], seq[m - 1 - i]))
            # ρ'' = (x1, x_{2k})(x2, x_{2k-1})...(x_k, x_{k+2})
            for i in range(1, k + 1):
                rho_double_prime.append(transposition(n, seq[i], seq[m - i]))
        return (
            TranspositionsList(list(filter(None, rho_prime))),
            TranspositionsList(list(filter(None, rho_double_prime))),
        )

    def __str__(self):
        return str(tuple(self.cycle_repr))


class Transposition(Permutation):
    def __init__(self, n: int, a: int, b: int):
        self._a = a
        self._b = b
        if a == b:
            mapping = list(range(n))
        else:
            mapping = list(range(n))
            mapping[a], mapping[b] = mapping[b], mapping[a]

        super().__init__(mapping)
        assert (
            len(self._cycles_in_tuple()) == 1 and len(self._cycles_in_tuple()[0]) == 2
        )

    @property
    def a(self) -> int:
        return self._a

    @property
    def b(self) -> int:
        return self._b

    def __len__(self) -> int:
        return 2


class TranspositionsList:
    """A class representing a list of transpositions."""

    def __init__(self, transpositions: List[Transposition]):
        assert len({t.n for t in transpositions}) == 1 or not transpositions
        self._transpositions = transpositions

    @property
    def transpositions(self) -> List[Transposition]:
        return self._transpositions

    def __len__(self) -> int:
        return len(self.transpositions)

    def __getitem__(self, index: int) -> Transposition:
        return self.transpositions[index]

    def __iter__(self):
        return iter(self.transpositions)

    def __repr__(self) -> str:
        return f"TranspositionsList({self.transpositions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TranspositionsList):
            return NotImplemented
        return self.transpositions == other.transpositions

    def __call__(self, x: int) -> int:
        for tau in self.transpositions:
            x = tau(x)
        return x

    def __add__(self, other: TranspositionsList) -> TranspositionsList:
        if not isinstance(other, TranspositionsList):
            return NotImplemented
        if len(self.transpositions) == 0:
            return other
        if len(other.transpositions) == 0:
            return self
        if self.transpositions[0].n != other.transpositions[0].n:
            raise ValueError("Transpositions must act on the same set")
        return TranspositionsList(self.transpositions + other.transpositions)

    @property
    def elements(self) -> List[int]:
        """
        Return the list of elements involved in the transpositions.
        from : (x0,x1)◦(x2,x3)◦···◦(x2m-2,x2m-1),
        returns [x0, x1, x2, x3, ..., x2m-2, x2m-1]
        """
        elems = []
        for t in self.transpositions:
            elems += sorted([t.a, t.b])
        return elems

    @property
    def n(self) -> int:
        return self.transpositions[0].n

    def apply_to_state(self, sv: np.ndarray) -> np.ndarray:
        reordered_sv = np.zeros_like(sv)
        for i in range(len(sv)):
            new_index = self(i)
            reordered_sv[new_index] = sv[i]
        return reordered_sv


def transformed_binary_matrix(
    binary_matrix: np.ndarray, qc: cirq.Circuit, qubits: List[cirq.LineQubit]
) -> np.ndarray:
    transformed = binary_matrix.copy()
    for row_idx, row in enumerate(transformed):
        in_int = int("".join(row.astype(str)), 2)
        qc_res = qc.final_state_vector(
            initial_state=cirq.one_hot(
                index=in_int, shape=2 ** len(qubits), dtype=np.complex128
            ),
            qubit_order=qubits,
        )
        nonzero = np.where(qc_res > 1e-8)[0]
        assert len(nonzero) == 1, f"State is not classical: qc_res={qc_res}"
        out_binary = list(map(int, format(nonzero[0], f"0{len(qubits)}b")))
        transformed[row_idx] = out_binary

    return transformed


class DisjointTranspositions(TranspositionsList):
    """
    A class representing a list of disjoint transpositions.

    This class models a specific type of transpositions list where the transpositions
    are applied in the form:
        (x0,x1)◦(x2,x3)◦···◦(x2m-2,x2m-1)
    (following the notation in https://arxiv.org/pdf/2406.16142)
    """

    def __init__(self, transpositions: List[Transposition]):
        transpose_elts = set()
        for t in transpositions:
            transpose_elts.add(t.a)
            transpose_elts.add(t.b)
        assert len(transpose_elts) == 2 * len(
            transpositions
        ), "Transpositions must be disjoint."

        super().__init__(transpositions)

        if transpositions:
            assert np.isclose(
                np.log2(len(self.elements)),
                round(np.log2(len(self.elements))),
            )

    @property
    def m(self) -> int:
        return len(self.transpositions)

    def induced_index_extraction_map(self) -> Permutation:
        mapping = list(range(0, self.n))
        num_qubits = int(np.log2(self.n))
        qubits = cirq.LineQubit.range(num_qubits)

        qc = self.index_extraction_map_qc(qubits, mcx_gate_type=CanonMCXGate)

        for i in range(self.n):
            in_ = cirq.one_hot(index=i, shape=2**num_qubits, dtype=np.complex128)
            qc_res = qc.final_state_vector(initial_state=in_, qubit_order=qubits)
            nonzero = np.where(qc_res > 1e-8)[0]
            assert len(nonzero) == 1, f"State is not classical: qc_res={qc_res}"
            mapping[i] = nonzero[0]

        return Permutation(mapping)

    def to_binary_matrix(self, binary_length: int) -> np.ndarray:
        binary_matrix = []
        for element in self.elements:
            binary_representation = format(element, f"0{binary_length}b")
            binary_row = [int(bit) for bit in binary_representation]
            binary_matrix.append(binary_row)
        return np.array(binary_matrix)

    def index_extraction_map_qc(
        self,
        qubits: List[cirq.LineQubit],
        mcx_gate_type: Type[MCXGateBase],
        do_validation: bool = False,
    ) -> Circuit:
        """
        The algorithm in https://arxiv.org/abs/2406.16142
        for constructing σi,1 in quantum circuit.
        """
        qc = cirq.Circuit()

        permuted_qubits = sorted(list(reversed((qubits)))[: int(log2(2 * self.m))])
        permuted_indices = [qubits.index(q) for q in permuted_qubits]
        non_permuted_qubits = qubits[: len(qubits) - int(log2(2 * self.m))]
        non_permuted_inidices = [qubits.index(q) for q in non_permuted_qubits]

        binary_matrix = self.to_binary_matrix(len(qubits))

        # process first row only, by using X gates
        first_row = binary_matrix[0]
        for idx, e in enumerate(first_row):
            if e == 1:
                qc.append(cirq.X(qubits[idx]))
        binary_matrix = transformed_binary_matrix(
            binary_matrix=binary_matrix, qc=qc, qubits=qubits
        )

        def contains_one_in_non_permute_qubit(row: Sequence[int]):
            ret = list()
            for i in non_permuted_inidices:
                if row[i] == 1:
                    ret.append(i)
            return ret

        for idx in range(1, len(binary_matrix)):
            row = binary_matrix[idx]
            target_row_bits = format(idx, f"0{len(qubits)}b")
            target_row = [int(bit) for bit in target_row_bits]
            if non_zero_non_permut_indicies := contains_one_in_non_permute_qubit(row):
                for i in permuted_indices:
                    if target_row[i] != row[i]:
                        qc.append(
                            cirq.CX(qubits[non_zero_non_permut_indicies[0]], qubits[i])
                        )
                non_zero_indices = [i for i, bit in enumerate(target_row) if bit == 1]
                for i in non_zero_non_permut_indicies:
                    mcx_main_qubits = [qubits[j] for j in non_zero_indices] + [
                        qubits[i]
                    ]
                    available_free_qubits = [
                        q for q in qubits if q not in mcx_main_qubits
                    ]

                    qc.append(
                        cirq.decompose_once(
                            mcx_gate_type.from_available_aux_qubits(
                                mcx_main_qubits, available_free_qubits
                            )
                        )
                    )
            else:
                non_zero_permute_qubits = [
                    qubits[i] for i in permuted_indices if row[i] == 1
                ]
                mcx_main_qubits = non_zero_permute_qubits + [non_permuted_qubits[0]]
                available_free_qubits = [q for q in qubits if q not in mcx_main_qubits]
                qc.append(
                    cirq.decompose_once(
                        mcx_gate_type.from_available_aux_qubits(
                            mcx_main_qubits, available_free_qubits
                        )
                    )
                )

                for i in permuted_indices:
                    if target_row[i] != row[i]:
                        qc.append(cirq.CX(non_permuted_qubits[0], qubits[i]))
                non_zero_indices = [i for i, bit in enumerate(target_row) if bit == 1]

                mcx_main_qubits = [qubits[j] for j in non_zero_indices] + [
                    non_permuted_qubits[0]
                ]
                free_qubits = [q for q in qubits if q not in mcx_main_qubits]

                qc.append(
                    cirq.decompose_once(
                        mcx_gate_type.from_available_aux_qubits(
                            mcx_main_qubits, free_qubits
                        )
                    )
                )
            binary_matrix = transformed_binary_matrix(
                binary_matrix=self.to_binary_matrix(len(qubits)), qc=qc, qubits=qubits
            )

        if do_validation:
            # Validation
            for idx, x in enumerate(self.elements):
                sv = cirq.one_hot(index=x, shape=2 ** len(qubits), dtype=np.complex128)
                qc_res = qc.final_state_vector(initial_state=sv, qubit_order=qubits)
                nonzero = np.where(qc_res > 1e-8)[0]
                assert len(nonzero) == 1
                assert (
                    nonzero[0] == idx
                ), f"should be mapped to idx {idx} but got {nonzero[0]}"

        return qc

    def __repr__(self) -> str:
        return f"DisjointTranspositions({self.transpositions})"


class SequentialTranspositions(TranspositionsList):
    """
    A class representing a list of transpositions applied sequentially.

    This class models a specific type of transpositions list where the transpositions
    are applied in the form:
        (0,1)◦(2,3)◦···◦(2m-2,2m-1)
    ($m$ is for following the notation in https://arxiv.org/pdf/2406.16142)
    """

    def __init__(self, transpositions: List[Transposition]):
        for idx, t in enumerate(transpositions):
            assert t.a + 1 == t.b or t.a == t.b + 1
            if idx + 1 != len(transpositions):
                assert max(t.a, t.b) + 1 == min(
                    transpositions[idx + 1].a, transpositions[idx + 1].b
                )
        super().__init__(transpositions)

    @classmethod
    def from_num_transposes(
        cls, n: int, num_transposes: int
    ) -> SequentialTranspositions:
        """
        Create a SequentialTranspositions object with transpositions
        (0,1)◦(2,3)◦···◦(2m-2,2m-1) where m = num_transposes
        """
        if not (np.log2(num_transposes).is_integer() and num_transposes >= 1):
            raise ValueError("num_transposes must be a power of 2.")

        transpositions = [
            Transposition(n, 2 * i, 2 * i + 1) for i in range(0, num_transposes)
        ]

        return cls(transpositions)

    def to_quantum_circuit(
        self,
        qubits: List[cirq.LineQubit],
        mcx_gate_type: Type[MCXGateBase],
        do_validation: bool = False,
    ) -> Circuit:
        max_elt_in_transpositions = max([max(t.a, t.b) for t in self.transpositions])
        targ_qubit_range = math.ceil(log2(max_elt_in_transpositions))

        assert np.isclose(
            np.log2(max_elt_in_transpositions + 1),
            round(np.log2(max_elt_in_transpositions + 1)),
        ), "max_elt_in_transpositions + 1 must be a power of two"
        assert targ_qubit_range < len(qubits)
        assert max_elt_in_transpositions % 2 == 1
        assert 2 ** len(qubits) >= max_elt_in_transpositions

        qc = Circuit()
        if max_elt_in_transpositions == 1:
            target = qubits[-1]
            controls = qubits[: len(qubits) - 1]

            main_mcx_qubits = controls + [target]
            free_qubits = [q for q in qubits if q not in main_mcx_qubits]

            qc.append(
                cirq.decompose_once(
                    mcx_gate_type.from_available_aux_qubits(
                        main_mcx_qubits, free_qubits, control_values=[0] * len(controls)
                    )
                )
            )
        else:
            raise NotImplementedError("Possibly a bug in following.")
            target = qubits[-1]
            controls = qubits[: len(qubits) - targ_qubit_range]

            qc.append(
                cirq.X.controlled(
                    num_controls=len(controls), control_values=[0] * len(controls)
                ).on(*controls, target)
            )

        # Validation
        if do_validation:
            for x in range(self.n):
                sv = cirq.one_hot(index=x, shape=2 ** len(qubits), dtype=np.complex128)
                qc_res = qc.final_state_vector(initial_state=sv, qubit_order=qubits)
                nonzero = np.where(qc_res > 1e-8)[0]
                assert len(nonzero) == 1
                assert nonzero[0] == self(x)

        return qc

    def __repr__(self) -> str:
        return f"SequentialTranspositions({self.transpositions})"
