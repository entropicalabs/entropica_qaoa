"""
Test that all the components of qaoa play nicely together
"""
import os, sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import pytest
import numpy as np

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, local_qvm, get_qc

from vqe.optimizer import scipy_optimizer
from qaoa.cost_function import QAOACostFunctionOnWFSim, QAOACostFunctionOnQVM
from qaoa.parameters import FourierQAOAParameters


@pytest.mark.slow
def test_qaoa_on_wfsim():
    hamiltonian = PauliSum.from_compact_str("(-1.0)*Z0*Z1 + 0.8*Z0 + (-0.5)*Z1")
    params = FourierQAOAParameters.from_hamiltonian(hamiltonian, timesteps=10, q=2)
    p0 = params.raw()
    sim=WavefunctionSimulator()
    log =[]
    cost_fun = QAOACostFunctionOnWFSim(hamiltonian, params, sim,
                                       noisy=True,
                                       log=log)
    with local_qvm():
        out = scipy_optimizer(cost_fun, params0=p0, epsilon=1e-3,
                              maxiter=500)
        wf = sim.wavefunction(cost_fun.prepare_ansatz,
                              memory_map=cost_fun.make_memory_map(params))
    assert np.allclose(out["fun"], -1.3, rtol=1.1)
    assert out["success"]
    assert np.allclose(np.abs(wf.amplitudes**2), [0, 0, 0, 1], rtol=1.5, atol=0.05)
    print(out)


@pytest.mark.slow
def test_qaoa_on_qvm():
    hamiltonian = PauliSum.from_compact_str("(-1.0)*Z0*Z1 + 0.8*Z0 + (-0.5)*Z1")
    params = FourierQAOAParameters.from_hamiltonian(hamiltonian, timesteps=10, q=2)
    p0 = params.raw()
    qvm = get_qc("2q-qvm")
    log =[]
    with local_qvm():
        cost_fun = QAOACostFunctionOnQVM(hamiltonian, params, qvm, base_numshots=50)
        out = scipy_optimizer(cost_fun, params0=p0, nshots=4, epsilon=2e-1,
                              maxiter=100)
    assert np.allclose(out["fun"], -1.3, rtol=1.1)
    assert out["success"]
    print(out)