import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np

from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil import get_qc, Program
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder

from qaoa.cost_function import QAOACostFunctionOnQVM, QAOACostFunctionOnWFSim
from qaoa.parameters import AdiabaticTimestepsQAOAParameters,\
    AlternatingOperatorsQAOAParameters

# Create a mixed and somehwat more complicated hamiltonian
# TODO fix the whole Qubit Placeholder Business
hamiltonian = PauliSum.from_compact_str("1.0*Z0Z1")
hamiltonian += PauliTerm("Z", 1, 0.5)
next_term = PauliTerm("Z", 0, -2.0)
next_term *= PauliTerm("Z", 1)
hamiltonian += next_term

# TODO verfiy, that the results actually make sense


def test_QAOACostFunctionOnWFSim():
    sim = WavefunctionSimulator()
    log = []
    params = AdiabaticTimestepsQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, timesteps=4)

    with local_qvm():
        cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                                params=params,
                                                sim=sim,
                                                scalar_cost_function=False,
                                                noisy=True,
                                                log=log)
        out = cost_function(params.raw(), nshots=100)
        print("output of QAOACostFunctionOnWFSim: ", out)


def test_QAOACostFunctionOnWFSim_get_wavefunction():
    sim = WavefunctionSimulator()
    ham = PauliSum.from_compact_str("0.7*Z0*Z1 + 1.2*Z0*Z2")
    timesteps = 2
    params = AlternatingOperatorsQAOAParameters\
        .linear_ramp_from_hamiltonian(ham, timesteps)
    with local_qvm():
        cost_function = QAOACostFunctionOnWFSim(ham,
                                                params=params,
                                                sim=sim,
                                                scalar_cost_function=True,
                                                nshots=100)
        wf = cost_function.get_wavefunction(params.raw())
        print(wf.probabilities())
        assert np.allclose(wf.probabilities(),
                           np.array([0.01, 0.308, 0.053, 0.13,
                            0.13, 0.053, 0.308, 0.01]),
                           rtol=1e-2, atol=0.005)


def test_QAOACostFunctionOnQVM():
    qvm = get_qc("2q-qvm")
    log = []
    params = AdiabaticTimestepsQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian, timesteps=4)

    with local_qvm():
        cost_function = QAOACostFunctionOnQVM(hamiltonian,
                                              params=params,
                                              qvm=qvm,
                                              scalar_cost_function=False,
                                              log=log)
        out = cost_function(params.raw(), nshots=1)
        print("output of QAOACostFunctionOnQVM: ", out)
