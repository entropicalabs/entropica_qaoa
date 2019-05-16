import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder, Qubit

from qaoa.parameters import GeneralQAOAParameters,\
    AlternatingOperatorsQAOAParameters, AdiabaticTimestepsQAOAParameters,\
    FourierQAOAParameters, QAOAParameterIterator, AbstractQAOAParameters

# build a hamiltonian to test everything on
q1 = QubitPlaceholder()
hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian += PauliTerm("Z", q1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", q1)
hamiltonian += next_term

# TODO Test plot functionality
# TODO test fourier params
# TODO Test set_hyperparameters and update_variable_parameters
def test_GeneralQAOAParameters():
    params = GeneralQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == {0, 1, q1}
    assert np.allclose(params.betas, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.gammas_singles, [[0.125], [0.375]])
    assert np.allclose(params.gammas_pairs, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())

# TODO check that the values also make sense
def test_GeneralQAOAParametersfromAbstractParameters():
    abstract_params = AbstractQAOAParameters((hamiltonian, 2))
    betas          = [[0.0, 0.1, 0.3], [0.5, 0.2, 1.2]]
    gammas_singles = [[0.0], [0.5]]
    gammas_pairs   = [[0.1, 0.3], [0.2, 1.2]]
    parameters = (betas, gammas_singles, gammas_pairs)
    general_params = GeneralQAOAParameters.from_AbstractParameters(abstract_params, parameters)
    print("The rotation angles from GeneralQAOAParameters.fromAbstractParameters")
    print("x_rotation_angles:\n", general_params.x_rotation_angles)
    print("z_rotation_angles:\n", general_params.z_rotation_angles)
    print("zz_rotation_angles:\n", general_params.zz_rotation_angles)

# Todo: Check that the values also make sense
def test_AlternatingOperatorsQAOAParametersfromAbstractParameters():
    abstract_params = AbstractQAOAParameters((hamiltonian, 2))
    betas          = [np.pi, 0.4]
    gammas_singles = [10, 24]
    gammas_pairs   = [8.8, 2.3]
    parameters = (betas, gammas_singles, gammas_pairs)
    alternating_params = AlternatingOperatorsQAOAParameters.from_AbstractParameters(abstract_params, parameters)
    print("The rotation angles from AlternatingOperatorsQAOAParameters.fromAbstractParameters")
    print("x_rotation_angles:\n", alternating_params.x_rotation_angles)
    print("z_rotation_angles:\n", alternating_params.z_rotation_angles)
    print("zz_rotation_angles:\n", alternating_params.zz_rotation_angles)
    assert type(alternating_params) == AlternatingOperatorsQAOAParameters


# Todo: Check that the values also make sense
def test_AdiabaticTimestepsQAOAParametersfromAbstractParameters():
    abstract_params = AbstractQAOAParameters((hamiltonian, 2))
    times = [0.4, 1.0]
    parameters = (times)
    adiabatic_params = AdiabaticTimestepsQAOAParameters.from_AbstractParameters(abstract_params, parameters, time=5.0)
    print("The rotation angles from AdiabaticTimestepsQAOAParameters.fromAbstractParameters")
    print("x_rotation_angles:\n", adiabatic_params.x_rotation_angles)
    print("z_rotation_angles:\n", adiabatic_params.z_rotation_angles)
    print("zz_rotation_angles:\n", adiabatic_params.zz_rotation_angles)
    assert type(adiabatic_params) == AdiabaticTimestepsQAOAParameters


# Todo: Check that the values also make sense
def test_FourierTimestepsQAOAParametersfromAbstractParameters():
    abstract_params = AbstractQAOAParameters((hamiltonian, 2))
    v = [0.4, 1.0]
    u_singles = [0.5, 1.2]
    u_pairs = [4.5, 123]
    parameters = (v, u_singles, u_pairs)
    fourier_params = FourierQAOAParameters.from_AbstractParameters(abstract_params, parameters, q=2)
    print("The rotation angles from AdiabaticTimestepsQAOAParameters.fromAbstractParameters")
    print("x_rotation_angles:\n", fourier_params.x_rotation_angles)
    print("z_rotation_angles:\n", fourier_params.z_rotation_angles)
    print("zz_rotation_angles:\n", fourier_params.zz_rotation_angles)
    assert type(fourier_params) == FourierQAOAParameters


def test_AdiabaticTimestepsQAOAParameters():
    params = AdiabaticTimestepsQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == {0, 1, q1}
    assert np.allclose(params.x_rotation_angles, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.z_rotation_angles, [[0.125], [0.375]])
    assert np.allclose(params.zz_rotation_angles, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_AlternatingOperatorsQAOAParameters():
    params = AlternatingOperatorsQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == {0, 1, q1}
    assert np.allclose(params.x_rotation_angles, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.z_rotation_angles, [[0.125], [0.375]])
    assert np.allclose(params.zz_rotation_angles, [[0.25, -0.5], [0.75, -1.5]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_FourierQAOAParameters():
    params = FourierQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, timesteps=3, q=2, time=2)
    # just access the angles, to check that it actually creates them
    assert len(params.z_rotation_angles) == len(params.zz_rotation_angles)
    assert np.allclose(params.v, [2/3, 0])
    assert np.allclose(params.u_singles, [2/3, 0])
    assert np.allclose(params.u_pairs, [2/3, 0])
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_QAOAParameterIterator():
    params = AdiabaticTimestepsQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, 2)
    iterator = QAOAParameterIterator(params, "times[0]", np.arange(0,1,0.5))
    log = []
    for p in iterator:
        log.append((p.times).copy())
    print(log[0])
    print(log[1])
    assert np.allclose(log[0], [0, 1.049999999])
    assert np.allclose(log[1], [0.5, 1.049999999])
