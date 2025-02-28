import cirq
import cirq.circuits

from state_preparation.state_prep_algorithms import LowRankStatePrep


def test():

    qc = cirq.Circuit()
    a, b, c = cirq.LineQubit.range(3)

    qc.append(cirq.H(a))
    qc.append(cirq.CX(a, b))
    qc.append(cirq.CX(a, c))

    sv = cirq.final_state_vector(qc)

    qc = LowRankStatePrep().run(sv)
    print(qc)
