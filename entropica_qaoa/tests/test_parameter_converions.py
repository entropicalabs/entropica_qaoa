import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from pytest import raises
import numpy as np

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder, Qubit

from entropica_qaoa.qaoa.parameters import (ExtendedParams,
                                            StandardWithBiasParams,
                                            AnnealingParams,
                                            FourierParams,
                                            FourierWithBiasParams,
                                            AbstractParams,
                                            StandardParams,)
from entropica_qaoa.qaoa._parameter_conversions import *


# build a hamiltonian to test everything on
q1 = QubitPlaceholder()
# hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian = PauliTerm("Z", 0)
hamiltonian *= PauliTerm("Z", 1)
hamiltonian += PauliTerm("Z", q1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", q1)
hamiltonian += next_term


def test_annealing_to_standard():
    params = AnnealingParams.linear_ramp_from_hamiltonian(hamiltonian,
                                                          2, time=2)
    params2 = annealing_to_standard(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_standard_to_standard_with_bias():
    params = StandardParams.empty((hamiltonian, 4))
    params2 = standard_to_standard_w_bias(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_standard_to_standard_with_bias():
    params = StandardWithBiasParams.empty((hamiltonian, 4))
    params2 = standard_w_bias_to_extended(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_fourier_to_standard():
    params = FourierParams.empty((hamiltonian, 4, 3))
    params2 = fourier_to_standard(params)
    assert isinstance(params2, StandardParams)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_fourier_w_bias_to_standard_w_bias():
    params = FourierWithBiasParams.empty((hamiltonian, 4, 3))
    params2 = fourier_w_bias_to_standard_w_bias(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_fourier_to_fourier_with_bias():
    params = FourierParams.empty((hamiltonian, 4, 3))
    params2 = fourier_to_fourier_w_bias(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_fourier_w_bias_to_fourier_extended():
    params = FourierWithBiasParams.empty((hamiltonian, 4, 3))
    params2 = fourier_w_bias_to_fourier_extended(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)


def test_fourier_extended_to_extended():
    params = FourierExtendedParams.empty((hamiltonian, 4, 3))
    params2 = fourier_extended_to_extended(params)
    assert np.allclose(params.x_rotation_angles, params2.x_rotation_angles)
    assert np.allclose(params.z_rotation_angles, params2.z_rotation_angles)
    assert np.allclose(params.zz_rotation_angles, params2.zz_rotation_angles)
