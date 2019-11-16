"""
Test functionality in entropica_qaoa.vqe.measurelib
"""
import numpy as np

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import RX, RY, H, CNOT
from pyquil.quil import Program, QubitPlaceholder, MEASURE
from pyquil.unitary_tools import lifted_pauli

from entropica_qaoa.vqe.measurelib import (append_measure_register,
                                           sampling_expectation_z_base,
                                           sampling_expectation,
                                           commuting_decomposition,
                                           base_change_fun,
                                           kron_eigs,
                                           wavefunction_expectation)


def test_commuting_decomposition():
    term1 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
    term2 = PauliTerm("Z", 1) * PauliTerm("Z", 2)
    term3 = PauliTerm("X", 0) * PauliTerm("X", 2)
    ham = PauliSum([term1, term2, term3])
    hams = commuting_decomposition(ham)
    assert hams == [ PauliSum([term3]), PauliSum([term1, term2])]


# TODO make a more complicated test case and sure, that the test case is
# actually correct
def test_sampling_expectation_z_base():
    bitstrings = np.array([[1,0], [0,1], [1,1], [1,1]])
    # ham = PauliSum.from_compact_str("1.0*Z0*Z1 + 0.5*Z0 + (-1)*Z1")
    term1 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
    term2 = PauliTerm("Z", 0, 0.5)
    term3 = PauliTerm("Z", 1, -1)
    ham = PauliSum([term1, term2, term3])
    out = sampling_expectation_z_base(ham, bitstrings)
    assert np.allclose(out, (0.25, 0.8958333333333334))


# TODO A more elaborate test?
def test_append_measure_register():
    q0 = QubitPlaceholder()
    p = Program(H(q0), RX(np.pi/2, 0))
    p = append_measure_register(p)
    assert str(p[-1]) == "MEASURE 0 ro[1]"


def test_sampling_expectation():
    bitstring1 = np.array([[1, 0], [0, 1], [1, 1], [1, 1]])
    bitstring2 = np.array([[1, 0], [0, 1], [1, 1], [1, 1]])
    bitstrings = [bitstring1, bitstring2]
    # ham1 = PauliSum.from_compact_str("1.0*Z0*Z1 + 0.5*Z0 + (-1)*Z1")
    # ham2 = PauliSum.from_compact_str("1.0*X0*Z1 + 0.5*X0 + (-1)*Z1")
    term1 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
    term2 = PauliTerm("Z", 0, 0.5)
    term3 = PauliTerm("Z", 1, -1)
    term4 = PauliTerm("X", 0) * PauliTerm("Z", 1)
    term5 = PauliTerm("X", 0, 0.5)
    ham1 = PauliSum([term1, term2, term3])
    ham2 = PauliSum([term4, term5, term3])
    hams = [ham1, ham2]
    out = sampling_expectation(hams, bitstrings)
    assert np.allclose(out, (0.5, 1.3385315336840842))


def test_base_change_fun():
    # this also tests apply_H and apply_RX
    term1 = PauliTerm("Z", 0) * PauliTerm("Z", 1)
    term2 = PauliTerm("Z", 1) * PauliTerm("Z", 2)
    term3 = PauliTerm("X", 0) * PauliTerm("X", 2)
    ham = PauliSum([term1, term2, term3])
    wf = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    wfs = []
    wfs.append(np.array([5, -1, 9, -1, -4, 0, -4, 0]))
    wfs.append(np.array([0, 1, 2, 3, 4, 5, 6, 7]))
    hams = commuting_decomposition(ham)
    for w, h in zip(wfs, hams):
        fun = base_change_fun(h, [0, 1, 2])
        assert np.allclose(fun(wf), w)


def test_kron_diagonal():
    term1 = PauliTerm("Z", 1, .2) * PauliTerm("Z", 3)
    term2 = PauliTerm("Z", 1, -.4) * PauliTerm("Z", 2)
    term3 = PauliTerm("Z", 1, 1.4)
    ham = PauliSum([term1, term2, term3])
    diag1 = np.diag(lifted_pauli(ham, [1, 3, 2]))
    diag2 = kron_eigs(ham, [1, 3, 2])
    assert np.allclose(diag1, diag2)


def test_wavefunction_expectation():
    term1 = PauliTerm("Z", 1, .2) * PauliTerm("Z", 2)
    term2 = PauliTerm("X", 1, -.4) * PauliTerm("Z", 3)
    term3 = PauliTerm("Y", 1, 1.4)
    ham = PauliSum([term1, term2, term3])
    ham2 = ham * ham

    mat = lifted_pauli(ham, [1, 3, 2])
    mat2 = lifted_pauli(ham2, [1, 3, 2])
    wf = np.array([0, 1.0, 2, .3, .5, -.5, .6, -.9])

    hams = commuting_decomposition(ham)
    hams2 = commuting_decomposition(ham2)

    funs = [base_change_fun(ham, [1, 3, 2]) for ham in hams]
    funs2 = [base_change_fun(ham, [1, 3, 2]) for ham in hams2]

    hams = [kron_eigs(h, [1, 3, 2]) for h in hams]
    hams2 = [kron_eigs(h, [1, 3, 2]) for h in hams2]

    e1, s1 = wavefunction_expectation(hams, funs, hams2, funs2, wf)
    e2, s2 = wf.conj()@mat@wf, wf@mat2@wf
    print(e1, s1)
    print(e2, s2)
    assert np.allclose((e1, s1), (e2, s2))
