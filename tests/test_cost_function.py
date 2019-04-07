import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from vqe.cost_function import prep_and_measure_ham_qvm
from qvm_process import qvm_process

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import RX, RY, H, CNOT
from pyquil.quil import Program

import numpy as np

# start a qvm for the tests
proc = qvm_process("9q-qvm")
qvm = proc.qvm
sim = proc.sim

def test_prep_and_measure_ham_qvm():
    def prepare_ansatz(params):
        p = Program()
        p.inst(RX(params[0] ,0))
        p.inst(RX(params[1], 1))
        return p

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    cost_fn = prep_and_measure_ham_qvm(prepare_ansatz, ham, sim, log=log)
    out = cost_fn([np.pi, np.pi/2], nshots=100)
    assert np.allclose(log ,[(-1.0, 0.1)])

# and finally kill the qvm, after the tests ran
proc.kill()
