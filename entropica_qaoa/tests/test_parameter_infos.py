import os
import sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import numpy as np
import scipy.optimize

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.api import WavefunctionSimulator, local_qvm, get_qc
from pyquil.quil import Program
from pyquil.gates import RX, CNOT

from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnWFSim,
                                           PrepareAndMeasureOnQVM)
from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim
from entropica_qaoa.qaoa.parameters import ExtendedParams

# Define a simple Hamiltonian
hamiltonian = PauliSum([PauliTerm("Z", 0, -1.0) * PauliTerm("Z", 1, 1.0),
                        PauliTerm("Z", 1, -1.0) * PauliTerm("Z", 2, 2.0),
                        PauliTerm("Z", 0, 0.8),
                        PauliTerm("Z", 1, -0.5),
                        PauliTerm("Z", 2, 0.8)])

# Instantiate an (empty) initial state prep circuit
# prepare_ansatz = Program()


def test_parameter_infos():
    params = ExtendedParams.linear_ramp_from_hamiltonian(hamiltonian,
                                                         n_steps=2)
    print(params)
    p0 = params.raw()
    print(p0)
    sim = WavefunctionSimulator()
    cost_fun = QAOACostFunctionOnWFSim(hamiltonian=hamiltonian,
                                       params=params,
                                       sim=sim,
                                       scalar_cost_function=True,
                                       nshots=1,
                                       noisy=False)
    with local_qvm():
        out = scipy.optimize.minimize(cost_fun, p0, tol=1e-3, method="Cobyla")
        print(out)

# Now the user would need to take the optimal parameters and specify which they want to modify, the range, etc.
# We decided the best way to do this is by a dictionary
# Perhaps we return by default from the optimiser the parameters in the dict form, with an entry that identifies their type.
# Also, when the user creates the parameter set initially, we could create a dict that has as the keys the parameter name, and
# as values the index in the raw list.
# Then the user can specify the name in a pre-determined format, and we easily index into the dictionary where it should be.
