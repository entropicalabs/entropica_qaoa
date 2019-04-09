"""
Tests for all functions in cost_function.py
TODO Fix the relative import hack in the first few lines
"""

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np

from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil import get_qc, Program
from pyquil.gates import RX, CNOT
from pyquil.paulis import PauliSum
from vqe.cost_function import PrepareAndMeasureOnWFSim, PrepareAndMeasureOnQVM


def test_PrepareAndMeasureOnWFSim():
    def prepare_ansatz(params):
        p = Program()
        p.inst(RX(params[0], 0))
        p.inst(RX(params[1], 1))
        return p

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    sim = WavefunctionSimulator()
    with local_qvm():
        cost_fn = PrepareAndMeasureOnWFSim(prepare_ansatz, ham, sim, log=log)
        out = cost_fn([np.pi, np.pi / 2], nshots=100)
        assert np.allclose(log, [(-1.0, 0.1)])
        assert np.allclose(out, -1)


def PrepareAndMeasureOnQVM():
    prepare_ansatz = Program()
    param_register = prepare_ansatz.declare(
        "params", memory_type="REAL", memory_size=2)
    prepare_ansatz.inst(RX(param_register[0], 0))
    prepare_ansatz.inst(RX(param_register[1], 1))

    def make_memory_map(params):
        return {"params": params}

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    qvm = get_qc("2q-qvm")
    with local_qvm():
        #        qvm = proc.qvm
        cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map, qvm=qvm,
                                         hamiltonian=ham, log=log,
                                         return_standard_deviation=True,
                                         base_numshots=10)
        out = cost_fn([np.pi, np.pi / 2], nshots=10)
        assert np.allclose(log, [(-1.0, 0.1)], rtol=1.1)
        assert np.allclose(out, -1, rtol=1.1)
