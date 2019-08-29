import sys, os
import math
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
import matplotlib.pyplot as plt
from pytest import raises

from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim
from pyquil.api import WavefunctionSimulator

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder, Qubit

from entropica_qaoa.qaoa.parameters import (ExtendedParams,
                                            StandardWithBiasParams,
                                            AnnealingParams,
                                            FourierParams,
                                            FourierWithBiasParams,
                                            FourierExtendedParams,
                                            QAOAParameterIterator,
                                            AbstractParams,
                                            StandardParams,
                                            _is_iterable_empty)

# build a hamiltonian to test everything on
q1 = QubitPlaceholder()
# hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian = PauliTerm("Z", 0)
hamiltonian *= PauliTerm("Z", 1)
hamiltonian += PauliTerm("Z", q1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", q1)
hamiltonian += next_term

# TODO test fourier params
# TODO Test set_hyperparameters and update_variable_parameters


def test_is_iterable_empty():
    empty_lists = ([], [[]], [[],[]], [[], [[[]]]])
    empty_tuples = ((), (()), ((), ()), ((), (((())))))
    empty_arrays = (np.array(l) for l in empty_lists)

    for l in empty_lists:
        assert(_is_iterable_empty(l))

    for t in empty_tuples:
        assert(_is_iterable_empty(t))

    for a in empty_arrays:
        assert(_is_iterable_empty(a))


def test_ExtendedParams():
    params = ExtendedParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    assert set(params.reg) == {0, 1, q1}
    assert np.allclose(params.betas, [[0.75] * 3, [0.25] * 3])
    assert np.allclose(params.gammas_singles, [[0.25], [0.75]])
    assert np.allclose(params.gammas_pairs, [[0.25, 0.25], [0.75, 0.75]])
    assert [params.qubits_singles] == [term.get_qubits() for term in hamiltonian
                                       if len(term) == 1]
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


# TODO check that the values also make sense
def test_ExtendedParamsfromAbstractParameters():
    abstract_params = AbstractParams((hamiltonian, 2))
    betas          = [[0.0, 0.1, 0.3], [0.5, 0.2, 1.2]]
    gammas_singles = [[0.0], [0.5]]
    gammas_pairs   = [[0.1, 0.3], [0.2, 1.2]]
    parameters = (betas, gammas_singles, gammas_pairs)
    general_params = ExtendedParams.from_AbstractParameters(abstract_params, parameters)
    print("The rotation angles from ExtendedParams.fromAbstractParameters")
    print("x_rotation_angles:\n", general_params.x_rotation_angles)
    print("z_rotation_angles:\n", general_params.z_rotation_angles)
    print("zz_rotation_angles:\n", general_params.zz_rotation_angles)


# Todo: Check that the values also make sense
def test_StandardWithBiasParamsfromAbstractParameters():
    abstract_params = AbstractParams((hamiltonian, 2))
    betas          = [np.pi, 0.4]
    gammas_singles = [10, 24]
    gammas_pairs   = [8.8, 2.3]
    parameters = (betas, gammas_singles, gammas_pairs)
    alternating_params = StandardWithBiasParams.from_AbstractParameters(abstract_params, parameters)
    print("The rotation angles from StandardWithBiasParams.fromAbstractParameters")
    print("x_rotation_angles:\n", alternating_params.x_rotation_angles)
    print("z_rotation_angles:\n", alternating_params.z_rotation_angles)
    print("zz_rotation_angles:\n", alternating_params.zz_rotation_angles)
    assert type(alternating_params) == StandardWithBiasParams


# Todo: Check that the values also make sense
def test_AnnealingParamsfromAbstractParameters():
    abstract_params = AbstractParams((hamiltonian, 2))
    schedule = [0.4, 1.0]
    parameters = (schedule)
    adiabatic_params = AnnealingParams.from_AbstractParameters(abstract_params, parameters, time=5.0)
    print("The rotation angles from AnnealingParams.fromAbstractParameters")
    print("x_rotation_angles:\n", adiabatic_params.x_rotation_angles)
    print("z_rotation_angles:\n", adiabatic_params.z_rotation_angles)
    print("zz_rotation_angles:\n", adiabatic_params.zz_rotation_angles)
    assert type(adiabatic_params) == AnnealingParams


# Todo: Check that the values also make sense
def test_FourierParamsfromAbstractParameters():
    abstract_params = AbstractParams((hamiltonian, 2))
    v = [0.4, 1.0]
    u_singles = [0.5, 1.2]
    u_pairs = [4.5, 123]
    parameters = (v, u_singles, u_pairs)
    fourier_params = FourierParams.from_AbstractParameters(abstract_params, parameters, q=2)
    print("The rotation angles from AnnealingParams.fromAbstractParameters")
    print("x_rotation_angles:\n", fourier_params.x_rotation_angles)
    print("z_rotation_angles:\n", fourier_params.z_rotation_angles)
    print("zz_rotation_angles:\n", fourier_params.zz_rotation_angles)
    assert type(fourier_params) == FourierParams


# Todo: Check that the values also make sense
def test_FourierExtendedParamsfromAbstractParameters():
    abstract_params = AbstractParams((hamiltonian, 2))
    v = [[0.4]*3, [1.0]*3]
    u_singles = [[0.5], [1.2]]
    u_pairs = [[4.5]*2, [123]*2]
    parameters = (v, u_singles, u_pairs)
    fourier_params = FourierExtendedParams.from_AbstractParameters(abstract_params, parameters, q=2)
    print("The rotation angles from AnnealingParams.fromAbstractParameters")
    print("x_rotation_angles:\n", fourier_params.x_rotation_angles)
    print("z_rotation_angles:\n", fourier_params.z_rotation_angles)
    print("zz_rotation_angles:\n", fourier_params.zz_rotation_angles)
    assert type(fourier_params) == FourierExtendedParams


def test_AnnealingParams():
    params = AnnealingParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
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


def test_StandardWithBiasParams():
    params = StandardWithBiasParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
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

def test_StandardParams():
    params = StandardParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
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


def test_non_fourier_params_are_consistent():
    """
    Check that StandardParams, StandardWithBiasParams and
    ExtendedParams give the same rotation angles, given the same data"""
    p1 = StandardParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    p2 = ExtendedParams.linear_ramp_from_hamiltonian(hamiltonian, 2, time=2)
    p3 = StandardWithBiasParams.linear_ramp_from_hamiltonian(hamiltonian,
                                                             2, time=2)
    assert np.allclose(p1.x_rotation_angles, p2.x_rotation_angles)
    assert np.allclose(p2.x_rotation_angles, p3.x_rotation_angles)
    assert np.allclose(p1.z_rotation_angles, p2.z_rotation_angles)
    assert np.allclose(p2.z_rotation_angles, p3.z_rotation_angles)
    assert np.allclose(p1.zz_rotation_angles, p2.zz_rotation_angles)
    assert np.allclose(p2.zz_rotation_angles, p3.zz_rotation_angles)


def test_FourierParams():
    params = FourierParams.linear_ramp_from_hamiltonian(hamiltonian, n_steps=3, q=2, time=2)
    # just access the angles, to check that it actually creates them
    assert len(params.z_rotation_angles) == len(params.zz_rotation_angles)
    assert np.allclose(params.v, [1/3, 0])
    assert np.allclose(params.u, [1/3, 0])
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_FourierWithBiasParams():
    params = FourierWithBiasParams.linear_ramp_from_hamiltonian(hamiltonian, n_steps=3, q=2, time=2)
    # just access the angles, to check that it actually creates them
    assert len(params.z_rotation_angles) == len(params.zz_rotation_angles)
    assert np.allclose(params.v, [1/3, 0])
    assert np.allclose(params.u_singles, [1/3, 0])
    assert np.allclose(params.u_pairs, [1/3, 0])
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_FourierExtendedParams():
    params = FourierExtendedParams.linear_ramp_from_hamiltonian(hamiltonian, n_steps=3, q=2, time=2)
    # just access the angles, to check that it actually creates them
    assert len(params.z_rotation_angles) == len(params.zz_rotation_angles)
    assert np.allclose(params.v, [[1/3] * 3, [0] * 3])
    assert np.allclose(params.u_singles, [[1/3] * 1, [0] * 1])
    assert np.allclose(params.u_pairs, [[1/3] * 2, [0] * 2])
    # Test updating and raw output
    raw = np.random.rand(len(params))
    params.update_from_raw(raw)
    assert np.allclose(raw, params.raw())


def test_FourierParams_are_consistent():
    """
    Check, that both Fourier Parametrizations give the same rotation angles,
    given the same data"""
    params1 = FourierParams.linear_ramp_from_hamiltonian(
                  hamiltonian, n_steps=3, q=2, time=2)
    params2 = FourierWithBiasParams.linear_ramp_from_hamiltonian(
                  hamiltonian, n_steps=3, q=2, time=2)
    params3 = FourierExtendedParams.linear_ramp_from_hamiltonian(
                  hamiltonian, n_steps=3, q=2, time=2)

    assert np.allclose(params1.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params1.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params1.zz_rotation_angles, params2.zz_rotation_angles)
    assert np.allclose(params1.x_rotation_angles, params3.x_rotation_angles)
    assert np.allclose(params1.z_rotation_angles, params3.z_rotation_angles)
    assert np.allclose(params1.zz_rotation_angles, params3.zz_rotation_angles)


def test_parameter_empty():
    p = ExtendedParams.empty((hamiltonian, 4))
    assert isinstance(p, ExtendedParams)
    assert p.betas.shape == (4, 3)
    assert p.gammas_singles.shape == (4, 1)
    assert p.gammas_pairs.shape == (4, 2)

    p = StandardParams.empty((hamiltonian, 4))
    assert isinstance(p, StandardParams)
    assert p.betas.shape == (4,)
    assert p.gammas.shape == (4,)

    p = StandardWithBiasParams.empty((hamiltonian, 4))
    assert isinstance(p, StandardWithBiasParams)
    assert p.betas.shape == (4,)
    assert p.gammas_singles.shape == (4,)
    assert p.gammas_pairs.shape == (4,)

    p = AnnealingParams.empty((hamiltonian, 4, 2.0))
    assert isinstance(p, AnnealingParams)
    assert p.schedule.shape == (4,)

    p = FourierParams.empty((hamiltonian, 4, 2))
    assert isinstance(p, FourierParams)
    assert p.u.shape == (2,)
    assert p.v.shape == (2,)

    p = FourierWithBiasParams.empty((hamiltonian, 4, 2))
    assert isinstance(p, FourierWithBiasParams)
    assert p.u_singles.shape == (2,)
    assert p.u_pairs.shape == (2,)
    assert p.v.shape == (2,)

    p = FourierExtendedParams.empty((hamiltonian, 4, 2))
    assert isinstance(p, FourierExtendedParams)
    assert p.u_singles.shape == (2,1)
    assert p.u_pairs.shape == (2,2)
    assert p.v.shape == (2,3)


def test_QAOAParameterIterator():
    params = AnnealingParams.linear_ramp_from_hamiltonian(hamiltonian, 2)
    iterator = QAOAParameterIterator(params, "schedule[0]", np.arange(0,1,0.5))
    log = []
    for p in iterator:
        log.append((p.schedule).copy())
    print(log[0])
    print(log[1])
    assert np.allclose(log[0], [0, 0.75])
    assert np.allclose(log[1], [0.5, 0.75])


def test_inputChecking():
    # ham = PauliSum.from_compact_str("0.7*Z0*Z1")
    ham = PauliSum([PauliTerm("Z", 0, 0.7) * PauliTerm("Z", 1)])
    betas = [1, 2, 3, 4]
    gammas_singles = []
    gammas_pairs = [1, 2, 3]
    with raises(ValueError):
        params = ExtendedParams((ham, 3),
                                (betas, gammas_singles, gammas_pairs))


# Plot Tests

def test_StandardParams_plot():
    ham_no_bias = PauliTerm("Z", 0)
    ham_no_bias *= PauliTerm("Z", 1)
    next_term = PauliTerm("Z", 0, -2.0)
    next_term *= PauliTerm("Z", 2)
    ham_no_bias += next_term

    p = 5
    params = StandardParams.linear_ramp_from_hamiltonian(hamiltonian, p)
    fig, ax = plt.subplots()
    params.plot(ax=ax)
    # plt.show()

    p = 8
    params = StandardParams((hamiltonian, p),([0.1]*p, [0.2]*p))
    fig, ax = plt.subplots()
    params.plot(ax=ax)
    # plt.show()

    p = 2
    params = StandardParams((ham_no_bias,p),([5]*p, [10]*p))
    fig, ax = plt.subplots()
    params.plot(ax=ax)
    # plt.show()


def test_ExtendedParams_plot():
    ham_no_bias = PauliTerm("Z", 0)
    ham_no_bias *= PauliTerm("Z", 1)
    next_term = PauliTerm("Z", 0, -2.0)
    next_term *= PauliTerm("Z", 2)
    ham_no_bias += next_term

    p = 5
    params = ExtendedParams.linear_ramp_from_hamiltonian(ham_no_bias, p)
    fig, ax = plt.subplots()
    params.plot(ax=ax)
    # plt.show()

    p = 8
    params = ExtendedParams((ham_no_bias, p),
                            ([0.1] * p*len(ham_no_bias.get_qubits()),
                             [],
                             [0.2] * p*len(ham_no_bias)))
    fig, ax = plt.subplots()
    params.plot(ax=ax)
    # plt.show()


def test_extended_get_constraints():

    # TEST 1
    weights = [0.1, 0.3, 0.5, -0.7]
    ham = PauliSum.from_compact_str('{}*Z0*Z1 + {}*Z1*Z2 + {}*Z2*Z3 + {}*Z3*Z0'.format(*weights))
    p = 2

    params = ExtendedParams.linear_ramp_from_hamiltonian(ham, p)

    actual_constraints = params.get_constraints()

    expected_constraints = [(0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi),  # beta constraints
                            (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi),

                            (0, 2 * math.pi / weights[0]), (0, 2 * math.pi / weights[1]),  # gamma pair constraints
                            (0, 2 * math.pi / weights[2]), (0, 2 * math.pi / weights[3]),
                            (0, 2 * math.pi / weights[0]), (0, 2 * math.pi / weights[1]),
                            (0, 2 * math.pi / weights[2]), (0, 2 * math.pi / weights[3])]

    assert(np.allclose(expected_constraints, actual_constraints))

    cost_function = QAOACostFunctionOnWFSim(ham, params, sim=WavefunctionSimulator())
    np.random.seed(0)
    random_angles = np.random.uniform(-100, 100, size = len(params.raw()))
    value = cost_function(random_angles)

    normalised_angles = [random_angles[i] % actual_constraints[i][1] for i in range(len(params.raw()))]
    normalised_value = cost_function(normalised_angles)

    assert(np.allclose(value, normalised_value))

    # TEST 2
    weights = [0.1324, -0.32, 0.35, -0.7]
    bias_weight = 0.35
    ham = PauliSum.from_compact_str('{}*Z0 + {}*Z0*Z1 + {}*Z1*Z2 + {}*Z1*Z3 + {}*Z3*Z0'.format(bias_weight, *weights))
    p = 3

    params = ExtendedParams.linear_ramp_from_hamiltonian(ham, p)

    actual_constraints = params.get_constraints()

    expected_constraints = [(0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi),  # beta constraints
                            (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi),
                            (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi), (0, 2 * math.pi),

                            (0, 2*math.pi / bias_weight),  # gamma single constraints
                            (0, 2*math.pi / bias_weight),
                            (0, 2*math.pi / bias_weight),

                            (0, 2 * math.pi / weights[0]), (0, 2 * math.pi / weights[1]),  # gamma pair constraints
                            (0, 2 * math.pi / weights[2]), (0, 2 * math.pi / weights[3]),
                            (0, 2 * math.pi / weights[0]), (0, 2 * math.pi / weights[1]),
                            (0, 2 * math.pi / weights[2]), (0, 2 * math.pi / weights[3]),
                            (0, 2 * math.pi / weights[0]), (0, 2 * math.pi / weights[1]),
                            (0, 2 * math.pi / weights[2]), (0, 2 * math.pi / weights[3])]

    assert(np.allclose(expected_constraints, actual_constraints))

    cost_function = QAOACostFunctionOnWFSim(ham, params, sim=WavefunctionSimulator())
    np.random.seed(0)
    random_angles = np.random.uniform(-100, 100, size=len(params.raw()))
    value = cost_function(random_angles)

    normalised_angles = [random_angles[i] % actual_constraints[i][1] for i in range(len(params.raw()))]
    normalised_value = cost_function(normalised_angles)

    assert(np.allclose(value, normalised_value))

