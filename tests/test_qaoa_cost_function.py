import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil import get_qc, Program
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import QubitPlaceholder

from qaoa.cost_function import QAOACostFunctionOnQVM, QAOACostFunctionOnWFSim
from qaoa.parameters import AdiabaticTimestepsQAOAParameters

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
    params = AdiabaticTimestepsQAOAParameters.from_hamiltonian(hamiltonian, timesteps=4)

    with local_qvm():
        cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                                params=params,
                                                sim=sim,
                                                return_standard_deviation=True,
                                                noisy=True,
                                                log=log)
        out = cost_function(params.raw())
        print("output of QAOACostFunctionOnWFSim: ", out)

def test_QAOACostFunctionOnQVM():
    qvm = get_qc("2q-qvm")
    log = []
    params = AdiabaticTimestepsQAOAParameters.from_hamiltonian(hamiltonian, timesteps=4)

    with local_qvm():
        cost_function = QAOACostFunctionOnQVM(hamiltonian,
                                              params=params,
                                              qvm=qvm,
                                              return_standard_deviation=True,
                                              log=log)
        out = cost_function(params.raw())
        print("output of QAOACostFunctionOnQVM: ", out)
