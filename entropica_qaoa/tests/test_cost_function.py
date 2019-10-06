"""
Tests for all functions in cost_function.py
"""
import numpy as np

from pyquil.quil import (QubitPlaceholder,
                         get_default_qubit_mapping)
from pyquil.api import WavefunctionSimulator
from pyquil import get_qc, Program
from pyquil.gates import RX, RY, X
from pyquil.paulis import PauliSum, PauliTerm

from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnWFSim,
                                              PrepareAndMeasureOnQVM)


def test_PrepareAndMeasureOnWFSim():
    p = Program()
    params = p.declare("params", memory_type="REAL", memory_size=2)
    p.inst(RX(params[0], 0))
    p.inst(RX(params[1], 1))

    def make_memory_map(params):
        return {"params": params}

#    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    term1 = PauliTerm("Z", 0)
    term2 = PauliTerm("Z", 1)
    ham = PauliSum([term1, term2])
    sim = WavefunctionSimulator()
    cost_fn = PrepareAndMeasureOnWFSim(p,
                                       make_memory_map,
                                       ham,
                                       sim,
                                       scalar_cost_function=False,
                                       enable_logging=True)
    out = cost_fn([np.pi, np.pi / 2])
    print(cost_fn.log[0].fun)
    assert np.allclose(cost_fn.log[0].fun, (-1.0, 0.0))
    assert np.allclose(out, (-1, 0.0))


def test_PrepareAndMeasureOnWFSim_QubitPlaceholders():
    q1, q2 = QubitPlaceholder(), QubitPlaceholder()
    p = Program()
    params = p.declare("params", memory_type="REAL", memory_size=2)
    p.inst(RX(params[0], q1))
    p.inst(RX(params[1], q2))

    def make_memory_map(params):
        return {"params": params}

    ham = PauliSum([PauliTerm("Z", q1), PauliTerm("Z", q2)])
    qubit_mapping = get_default_qubit_mapping(p)
    sim = WavefunctionSimulator()
    cost_fn = PrepareAndMeasureOnWFSim(p, make_memory_map, ham, sim,
                                       enable_logging=True,
                                       qubit_mapping=qubit_mapping,
                                       scalar_cost_function=False,
                                       )
    out = cost_fn([np.pi, np.pi / 2])
    assert np.allclose(cost_fn.log[0].fun, (-1.0, 0.0))
    assert np.allclose(out, (-1, 0.0))


def test_PrepareAndMeasureOnQVM():
    prepare_ansatz = Program()
    param_register = prepare_ansatz.declare(
        "params", memory_type="REAL", memory_size=2)
    prepare_ansatz.inst(RX(param_register[0], 0))
    prepare_ansatz.inst(RX(param_register[1], 1))

    def make_memory_map(params):
        return {"params": params}

#    ham = PauliSum.from_compact_str("1.0*Z0 + 1.0*Z1")
    term1 = PauliTerm("Z", 0)
    term2 = PauliTerm("Z", 1)
    ham = PauliSum([term1, term2])
    qvm = get_qc("2q-qvm")
    cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map, qvm=qvm,
                                     hamiltonian=ham, enable_logging=True,
                                     scalar_cost_function=True,
                                     base_numshots=10,
                                     nshots=10)
    out = cost_fn([np.pi, np.pi / 2])
    assert np.allclose(cost_fn.log[0].fun, (-1.0, 0.1), rtol=1.1)
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
    qvm = get_qc("2q-qvm")
    cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map,
                                     qvm=qvm,
                                     hamiltonian=ham, enable_logging=True,
                                     scalar_cost_function=False,
                                     base_numshots=10,
                                     qubit_mapping=qubit_mapping)
    out = cost_fn([np.pi, np.pi / 2], nshots=10)
    assert np.allclose(cost_fn.log[0].fun, (-1.0, 0.1), rtol=1.1)
    assert np.allclose(out, (-1, 0.1), rtol=1.1)


def test_PrepareAndMeasureOnQVM_QubitPlaceholders_nondiag_hamiltonian():
    q1, q2, q3 = QubitPlaceholder(), QubitPlaceholder(), QubitPlaceholder()
    ham = PauliTerm("Y", q1)*PauliTerm("Z",q3)
    ham += PauliTerm("Y", q1)*PauliTerm("Z",q2,-0.3)
    ham += PauliTerm("Y", q1)*PauliTerm("X",q3, 2.0)
    params = [3.0,0.4,4.5]

    prepare_ansatz = Program()
    param_register = prepare_ansatz.declare(
        "params", memory_type="REAL", memory_size=3)
    prepare_ansatz.inst(RX(param_register[0], q1))
    prepare_ansatz.inst(RY(param_register[1], q2))
    prepare_ansatz.inst(RY(param_register[2], q3))

    def make_memory_map(params):
        return {"params": params}

    qubit_mapping = get_default_qubit_mapping(prepare_ansatz)
    qvm = get_qc("3q-qvm")
    cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map,
                                     qvm=qvm,
                                     hamiltonian=ham,
                                     scalar_cost_function=False,
                                     base_numshots=100,
                                     qubit_mapping=qubit_mapping)
    out = cost_fn(params, nshots=10)
    assert np.allclose(out, (0.346, 0.07), rtol=1.1)

def test_sample_bitstrings():
    
    """
    A simple circuit - test that it returns an easy bitstring
    Add a test like this to the test_qaoa_cost_function file too.
    """
    
    prepare_ansatz = Program()
    
    params = [np.pi]*3
    param_register = prepare_ansatz.declare(
    "params", memory_type="REAL", memory_size=3)
    
    for i in range(3):
        prepare_ansatz.inst(RX(param_register[i], i))
    
    # A simple hamiltonian
    # Note we need 3 qubits because the PrepareAndMeasureOnQVM adds a memory
    # bit for each qubit in the cost Hamiltonian. It also automatically applies
    # any relevant basis changes before measurements (because this is how we get expectation values), 
    # so here for simplicity we use just Z operators. The corresponding QAOA example is more illustrative.
    ham = PauliSum([PauliTerm("Z", 0), PauliTerm("Z", 1),PauliTerm("Z", 2)]) 
    
    def make_memory_map(params):
        return {"params": params}
    
    qvm = get_qc("3q-qvm")
    cost_fn = PrepareAndMeasureOnQVM(prepare_ansatz, make_memory_map,
                                     qvm=qvm,
                                     hamiltonian=ham,
                                     base_numshots=1)
    
    bitstring = cost_fn.sample_bitstrings(params, nshots=1)
    expected = np.array([1,1,1])
    
    assert np.array_equal(bitstring, expected)