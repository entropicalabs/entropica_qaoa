"""
Test the optimizer implementation
"""

import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, local_qvm, get_qc
from pyquil.quil import Program
from pyquil.gates import RX, CNOT

from entropica_qaoa.vqe.optimizer import scipy_optimizer
from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnWFSim,
                                           PrepareAndMeasureOnQVM)
from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim
from entropica_qaoa.qaoa.parameters import FourierParams

#hamiltonian = PauliSum.from_compact_str("(-1.0)*Z0*Z1 + 0.8*Z0 + (-0.5)*Z1")

hamiltonian = PauliSum([PauliTerm("Z", 0, -1.0) * PauliTerm("Z", 1, 1.0),
                        PauliTerm("Z", 0, 0.8), PauliTerm("Z", 1, -0.5)])
prepare_ansatz = Program()
params = prepare_ansatz.declare("params", memory_type="REAL", memory_size=4)
prepare_ansatz.inst(RX(params[0], 0))
prepare_ansatz.inst(RX(params[1], 1))
prepare_ansatz.inst(CNOT(0, 1))
prepare_ansatz.inst(RX(params[2], 0))
prepare_ansatz.inst(RX(params[3], 1))

p0 = [0, 0, 0, 0]


def test_vqe_on_WFSim():
    sim = WavefunctionSimulator()
    cost_fun = PrepareAndMeasureOnWFSim(prepare_ansatz=prepare_ansatz,
                                        make_memory_map=lambda p: {"params": p},
                                        hamiltonian=hamiltonian,
                                        sim=sim,
                                        return_standard_deviation=True,
                                        noisy=False)

    with local_qvm():
        out = scipy_optimizer(cost_fun, p0, epsilon=1e-3)
        print(out)
        wf = sim.wavefunction(prepare_ansatz, {"params": out['x']})
    assert np.allclose(np.abs(wf.amplitudes**2),
                       [0, 0, 0, 1],
                       rtol=1.5, atol=0.01)
    assert np.allclose(out['fun'], -1.3)
    assert out['success']


def test_qaoa_on_WFSim():
    params = FourierParams.linear_ramp_from_hamiltonian(hamiltonian,
                                                        n_steps=10, q=2)
    p0 = params.raw()
    sim = WavefunctionSimulator()
    cost_fun = QAOACostFunctionOnWFSim(hamiltonian=hamiltonian,
                                       params=params,
                                       sim=sim,
                                       return_standard_deviation=True,
                                       noisy=False)

    with local_qvm():
        out = scipy_optimizer(cost_fun, p0, epsilon=1e-3)
        print(out)
