import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder, Qubit

from qaoa.parameters import GeneralQAOAParameters,\
    AlternatingOperatorsQAOAParameters, AdiabaticTimestepsQAOAParameters,\
    FourierQAOAParameters, QAOAParameterIterator

# build a hamiltonian to test everything on
q1 = QubitPlaceholder()
hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian += PauliTerm("Z", q1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", q1)
hamiltonian += next_term

# TODO Test plot functionality
# TODO Test set_constant_parameters and update_variable_parameters
def test_GeneralQAOAParameters():
    params = GeneralQAOAParameters.from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == set([0, 1, q1])
    assert np.allclose(params.betas, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.gammas_singles, [[0.125], [0.375]])
    assert np.allclose(params.gammas_pairs, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update(raw)
    assert np.allclose(raw, params.raw())


def test_AdiabaticTimestepsQAOAParameters():
    params = AdiabaticTimestepsQAOAParameters.from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == set([0, 1, q1])
    assert np.allclose(params.betas, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.gammas_singles, [[0.125], [0.375]])
    assert np.allclose(params.gammas_pairs, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update(raw)
    assert np.allclose(raw, params.raw())

def test_AlternatingOperatorsQAOAParameters():
    params = AlternatingOperatorsQAOAParameters.from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == set([0, 1, q1])
    assert np.allclose(params.betas, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.gammas_singles, [[0.125], [0.375]])
    assert np.allclose(params.gammas_pairs, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update(raw)
    assert np.allclose(raw, params.raw())


def test_QAOAParameterIterator():
    params = AdiabaticTimestepsQAOAParameters.from_hamiltonian(hamiltonian, 2)
    iterator = QAOAParameterIterator(params, "_times[0]", np.arange(0,1,0.5))
    log = []
    for p in iterator:
        log.append((p._times).copy())
    print(log[0])
    print(log[1])
    assert np.allclose(log[0], [0, 1.049999999])
    assert np.allclose(log[1], [0.5, 1.049999999])
