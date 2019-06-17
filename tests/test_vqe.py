"""
Test that all the components of vqe play nicely together
"""

import os, sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from scipy.optimize import minimize
import pytest

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, local_qvm, get_qc
from pyquil.quil import Program
from pyquil.gates import RX, CNOT

from vqe.optimizer import scalar_cost_function
from vqe.cost_function import PrepareAndMeasureOnWFSim, PrepareAndMeasureOnQVM


# gonna need this program and hamiltonian for both tests. So define them globally
hamiltonian = PauliSum.from_compact_str("(-1.0)*Z0*Z1 + 0.8*Z0 + (-0.5)*Z1")
prepare_ansatz = Program()
params = prepare_ansatz.declare("params", memory_type="REAL", memory_size=4)
prepare_ansatz.inst(RX(params[0], 0))
prepare_ansatz.inst(RX(params[1], 1))
prepare_ansatz.inst(CNOT(0,1))
prepare_ansatz.inst(RX(params[2], 0))
prepare_ansatz.inst(RX(params[3], 1))

p0 = [0, 0, 0, 0]


@pytest.mark.slow
def test_vqe_on_WFSim():
    log = []
    sim = WavefunctionSimulator()
    cost_fun = PrepareAndMeasureOnWFSim(prepare_ansatz=prepare_ansatz,
                                        make_memory_map=lambda p: {"params": p},
                                        hamiltonian=hamiltonian,
                                        sim=sim,
                                        return_standard_deviation=True,
                                        noisy=False,
                                        log=log)
    fun = scalar_cost_function()(cost_fun)

    with local_qvm():
        out = minimize(fun, p0, tol=1e-3, method="COBYLA")
        print(out)
        wf = cost_fun.get_wavefunction(out['x'])
    assert np.allclose(wf.probabilities(), [0, 0, 0, 1], rtol=1.5, atol=0.01)
    assert np.allclose(out['fun'], -1.3)
    assert out['success']


@pytest.mark.slow
def test_vqe_on_QVM():
    p0 = [3.1, -1.5, 0, 0] # make it easier when sampling
    log = []
    qvm = get_qc("2q-qvm")
    with local_qvm():
        cost_fun = PrepareAndMeasureOnQVM(prepare_ansatz=prepare_ansatz,
                                          make_memory_map=lambda p: {"params": p},
                                          hamiltonian=hamiltonian,
                                          qvm=qvm,
                                          return_standard_deviation=True,
                                          base_numshots=50,
                                          log=log)
        fun = scalar_cost_function(nshots=4)(cost_fun)
        out = minimize(fun, p0, tol=1e-2, method="COBYLA")
        print(out)
    assert np.allclose(out['fun'], -1.3, rtol=1.1)
    assert out['success']
    print(out)
