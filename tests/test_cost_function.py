"""
Tests for all functions in cost_function.py
TODO Fix the relative import hack in the first few lines
"""

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np

from pyquil.quil import QubitPlaceholder, address_qubits, get_default_qubit_mapping
from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil import get_qc, Program
from pyquil.gates import RX, CNOT
from pyquil.paulis import PauliSum, PauliTerm
from vqe.cost_function import PrepareAndMeasureOnWFSim, PrepareAndMeasureOnQVM


def test_PrepareAndMeasureOnWFSim():
    p = Program()
    params = p.declare("params", memory_type="REAL", memory_size=2)
    p.inst(RX(params[0], 0))
    p.inst(RX(params[1], 1))

    def make_memory_map(params):
        return {"params": params}

    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    log = []
    sim = WavefunctionSimulator()
    with local_qvm():
        cost_fn = PrepareAndMeasureOnWFSim(p, make_memory_map,
                                           ham, sim, log=log)
        out = cost_fn([np.pi, np.pi / 2], nshots=100)
        assert np.allclose(log, [(-1.0, 0.1)])
        assert np.allclose(out, -1)


def test_PrepareAndMeasureOnWFSim_QubitPlaceholders():
    q1, q2 = QubitPlaceholder(), QubitPlaceholder()
    p = Program()
    params = p.declare("params", memory_type="REAL", memory_size=2)
    p.inst(RX(params[0], q1))
    p.inst(RX(params[1], q2))

    def make_memory_map(params):
        return {"params": params}

    ham = PauliSum([PauliTerm("Z", q1), PauliTerm("Z",q2)])
    qubit_mapping = get_default_qubit_mapping(p)
    log = []
    sim = WavefunctionSimulator()
    with local_qvm():
        cost_fn = PrepareAndMeasureOnWFSim(p, make_memory_map, ham, sim,
                                           log=log, qubit_mapping=qubit_mapping)
        out = cost_fn([np.pi, np.pi / 2], nshots=100)
        assert np.allclose(log, [(-1.0, 0.1)])
        assert np.allclose(out, -1)


def test_PrepareAndMeasureOnQVM():
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


def test_PrepareAndMeasureOnQVM_QubitPlaceholders():
    q1, q2 = QubitPlaceholder(), QubitPlaceholder()
    prepare_ansatz = Program()
    param_register = prepare_ansatz.declare(
        "params", memory_type="REAL", memory_size=2)
    prepare_ansatz.inst(RX(param_register[0], q1))
    prepare_ansatz.inst(RX(param_register[1], q2))

    def make_memory_map(params):
        return {"params": params}

    ham = PauliSum([PauliTerm("Z", q1), PauliTerm("Z",q2)])
    qubit_mapping = get_default_qubit_mapping(prepare_ansatz)
    log = []
    qvm = get_qc("2q-qvm")
    with local_qvm():
        #        qvm = proc.qvm
        cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map, qvm=qvm,
                                         hamiltonian=ham, log=log,
                                         return_standard_deviation=True,
                                         base_numshots=10,
                                         qubit_mapping=qubit_mapping)
        out = cost_fn([np.pi, np.pi / 2], nshots=10)
        assert np.allclose(log, [(-1.0, 0.1)], rtol=1.1)
        assert np.allclose(out, -1, rtol=1.1)
