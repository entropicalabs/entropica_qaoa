# import sys, os
# myPath = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, myPath + '/../')

# from vqe.cost_function import prep_and_measure_ham_qvm
# from qvm_process import qvm_process

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import RX, RY, H, CNOT
from pyquil.quil import Program, QubitPlaceholder, MEASURE

from vqe.measurelib import append_measure_register, hamiltonian_expectation_value

import numpy as np

# TODO make a more complicated test case and sure, that the test case is
# actually correct
def test_hamiltonian_expectation_value():
    bitstrings = np.array([[1,0], [0,1], [1,1], [1,1]])
    ham = PauliSum.from_compact_str("1.0*Z0*Z1 + 0.5*Z0 + (-1)*Z1")
    out = hamiltonian_expectation_value(ham, bitstrings)
    assert np.allclose(out, (0.25, 0.81967981553775))


# TODO A more elaborate test?
def test_append_measure_register():
    q0 = QubitPlaceholder()
    p = Program(H(q0), RX(np.pi/2, 0))
    p = append_measure_register(p)
    assert str(p[-1]) == "MEASURE 0 ro[1]"
