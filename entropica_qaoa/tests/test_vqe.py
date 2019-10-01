"""
Test that all the components of vqe play nicely together
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, get_qc
from pyquil.quil import Program
from pyquil.gates import RX, CNOT

from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnWFSim,
                                              PrepareAndMeasureOnQVM)


# gonna need this program and hamiltonian for both tests. So define them globally
# hamiltonian = PauliSum.from_compact_str("(-1.0)*Z0*Z1 + 0.8*Z0 + (-0.5)*Z1")
term1 = PauliTerm("Z", 0, -1)
term1 *= PauliTerm("Z", 1)
term2 = PauliTerm("Z", 0, 0.8)
term3 = PauliTerm("Z", 1, -0.5)
hamiltonian = PauliSum([term1, term2, term3])

prepare_ansatz = Program()
params = prepare_ansatz.declare("params", memory_type="REAL", memory_size=4)
prepare_ansatz.inst(RX(params[0], 0))
prepare_ansatz.inst(RX(params[1], 1))
prepare_ansatz.inst(CNOT(0, 1))
prepare_ansatz.inst(RX(params[2], 0))
prepare_ansatz.inst(RX(params[3], 1))

p0 = [0, 0, 0, 0]


@pytest.mark.slow
def test_vqe_on_WFSim():
    sim = WavefunctionSimulator()
    cost_fun = PrepareAndMeasureOnWFSim(prepare_ansatz=prepare_ansatz,
                                        make_memory_map=lambda p: {"params": p},
                                        hamiltonian=hamiltonian,
                                        sim=sim,
                                        scalar_cost_function=True)

    out = minimize(cost_fun, p0, tol=1e-3, method="COBYLA")
    wf = sim.wavefunction(prepare_ansatz, {"params": out['x']})
    assert np.allclose(wf.probabilities(), [0, 0, 0, 1], rtol=1.5, atol=0.01)
    assert np.allclose(out['fun'], -1.3)
    assert out['success']


@pytest.mark.slow
def test_vqe_on_QVM():
    p0 = [3.1, -1.5, 0, 0]  # make it easier when sampling
    qvm = get_qc("2q-qvm")
    cost_fun = PrepareAndMeasureOnQVM(prepare_ansatz=prepare_ansatz,
                                      make_memory_map=lambda p: {"params": p},
                                      hamiltonian=hamiltonian,
                                      qvm=qvm,
                                      scalar_cost_function=True,
                                      nshots=4,
                                      base_numshots=50)
    out = minimize(cost_fun, p0, tol=1e-2, method="Cobyla")
    assert np.allclose(out['fun'], -1.3, rtol=1.1)
    assert out['success']
