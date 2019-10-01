"""
Test the QAOA cost functions
"""

import numpy as np

from pyquil.api import WavefunctionSimulator
from pyquil import get_qc
from pyquil.paulis import PauliSum, PauliTerm

from entropica_qaoa.qaoa.cost_function import (QAOACostFunctionOnQVM,
                                               QAOACostFunctionOnWFSim)
from entropica_qaoa.qaoa.parameters import (AnnealingParams,
                                            StandardWithBiasParams)

# Create a mixed and somehwat more complicated hamiltonian
# TODO fix the whole Qubit Placeholder Business

# hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian = PauliTerm("Z", 1) * PauliTerm("Z", 1)
hamiltonian += PauliTerm("Z", 1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", 1)
hamiltonian += next_term

# TODO verfiy, that the results actually make sense


def test_QAOACostFunctionOnWFSim():
    sim = WavefunctionSimulator()
    params = AnnealingParams.linear_ramp_from_hamiltonian(hamiltonian, n_steps=4)

    cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                            params=params,
                                            sim=sim,
                                            scalar_cost_function=False,
                                            enable_logging=True)
    out = cost_function(params.raw(), nshots=100)
    print(out)


def test_QAOACostFunctionOnWFSim_get_wavefunction():
    sim = WavefunctionSimulator()
    # ham = PauliSum.from_compact_str("0.7*Z0*Z1 + 1.2*Z0*Z2")
    term1 = PauliTerm("Z", 0, 0.7) * PauliTerm("Z", 1)
    term2 = PauliTerm("Z", 0, 1.2) * PauliTerm("Z", 2)
    ham = PauliSum([term1, term2])
    timesteps = 2
    params = StandardWithBiasParams\
        .linear_ramp_from_hamiltonian(ham, timesteps)
    cost_function = QAOACostFunctionOnWFSim(ham,
                                            params=params,
                                            sim=sim,
                                            scalar_cost_function=True,
                                            nshots=100)
    wf = cost_function.get_wavefunction(params.raw())
    assert np.allclose(wf.probabilities(),
                       np.array([0.01, 0.308, 0.053, 0.13,
                                 0.13, 0.053, 0.308, 0.01]),
                       rtol=1e-2, atol=0.005)


def test_QAOACostFunctionOnQVM():
    qvm = get_qc("2q-qvm")
    params = AnnealingParams.linear_ramp_from_hamiltonian(hamiltonian, n_steps=4)

    cost_function = QAOACostFunctionOnQVM(hamiltonian,
                                          params=params,
                                          qvm=qvm,
                                          scalar_cost_function=False)
    out = cost_function(params.raw(), nshots=1)
    print(out)
