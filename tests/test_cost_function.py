import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from vqe.cost_function import prep_and_measure_ham_qvm, prep_and_measure_ham_qc
from qvm_process import qvm_process

from pyquil.paulis import PauliSum
from pyquil.gates import RX, CNOT
from pyquil.quil import Program

import numpy as np


# TODO fix the test sometimes randomly failing.
def test_prep_and_measure_ham_qvm():
    def prepare_ansatz(params):
        p = Program()
        p.inst(RX(params[0] ,0))
        p.inst(RX(params[1], 1))
        return p

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    with qvm_process("2q-qvm") as proc:
        sim = proc.sim
        cost_fn = prep_and_measure_ham_qvm(prepare_ansatz, ham, sim, log=log)
        print("pause")
        out = cost_fn([np.pi, np.pi / 2], nshots=100)
        assert np.allclose(log, [(-1.0, 0.1)])
        assert np.allclose(out, -1)


# TODO fix the test sometimes not running.
def test_prep_and_measure_ham_qc():
    prepare_ansatz = Program()
    param_register = prepare_ansatz.declare("params", memory_type="REAL", memory_size=2)
    prepare_ansatz.inst(RX(param_register[0] ,0))
    prepare_ansatz.inst(RX(param_register[1], 1))

    def make_memory_map(params):
        return {"params" : params}

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    with qvm_process("2q-qvm") as proc:
        qvm = proc.qvm
        sim = proc.sim
        cost_fn = prep_and_measure_ham_qc(prepare_ansatz, make_memory_map, qvm=qvm,
                                          hamiltonian=ham, log=log,
                                          return_standard_deviation=True,
                                          base_numshots=10)
        print("pause")
        out = cost_fn([np.pi, np.pi / 2], nshots=10)
        assert np.allclose(log, [(-1.0, 0.1)], rtol=1.1)
        assert np.allclose(out, -1, rtol=1.1)
