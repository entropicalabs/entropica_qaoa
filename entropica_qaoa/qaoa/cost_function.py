# Copyright 2019 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


"""
Implementation of the QAOA cost_functions. We inherit from
vqe/cost_functions and change only the QAOA specific details.
"""


from typing import Union, List, Type, Dict, Iterable
import numpy as np
from copy import deepcopy

from pyquil import Program
from pyquil.quil import MemoryReference, QubitPlaceholder, Qubit
from pyquil.wavefunction import Wavefunction
from pyquil.gates import RX, RZ, CPHASE, H
from pyquil.paulis import PauliSum
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.api._quantum_computer import QuantumComputer

from entropica_qaoa.vqe.cost_function import (PrepareAndMeasureOnQVM,
                                              PrepareAndMeasureOnWFSim,
                                              LogEntry)
from entropica_qaoa.qaoa.parameters import AbstractParams


def _qaoa_mixing_ham_rotation(betas: MemoryReference,
                              reg: Iterable) -> Program:
    """Produce parametric Quil-Code for the mixing hamiltonian rotation.

    Parameters
    ----------
    betas:
        Classic register to read the x_rotation_angles from.
    reg:
        The register to apply the X-rotations on.

    Returns
    -------
    Program
        Parametric Quil Program containing the X-Rotations.

    """
    if len(reg) != betas.declared_size:
        raise ValueError("x_rotation_angles must have the same length as reg")

    p = Program()
    for beta, qubit in zip(betas, reg):
        p.inst(RX(-2 * beta, qubit))
    return p


def _qaoa_cost_ham_rotation(gammas_pairs: MemoryReference,
                            qubit_pairs: List,
                            gammas_singles: MemoryReference,
                            qubit_singles: List) -> Program:
    """Produce the Quil-Code for the cost-hamiltonian rotation.

    Parameters
    ----------
    gammas_pairs:
        Classic register to read the zz_rotation_angles from.
    qubit_pairs:
        List of the Qubit pairs to apply rotations on.
    gammas_singles:
        Classic register to read the z_rotation_angles from.
    qubit_singles:
        List of the single qubits to apply rotations on.

    Returns
    -------
    Program
        Parametric Quil code containing the Z-Rotations.

    """
    p = Program()

    if len(qubit_pairs) != gammas_pairs.declared_size:
        raise ValueError("zz_rotation_angles must have the same length as qubits_pairs")

    for gamma_pair, qubit_pair in zip(gammas_pairs, qubit_pairs):
        p.inst(RZ(2 * gamma_pair, qubit_pair[0]))
        p.inst(RZ(2 * gamma_pair, qubit_pair[1]))
        p.inst(CPHASE(-4 * gamma_pair, qubit_pair[0], qubit_pair[1]))

    if gammas_singles.declared_size != len(qubit_singles):
        raise ValueError("z_rotation_angles must have the same length as qubit_singles")

    for gamma_single, qubit in zip(gammas_singles, qubit_singles):
        p.inst(RZ(2 * gamma_single, qubit))

    return p


def _qaoa_annealing_program(params: Type[AbstractParams]) -> Program:
    """Create parametric quil code for the QAOA annealing circuit.

    Parameters
    ----------
    params:
        The parameters of the QAOA circuit.

    Returns
    -------
    Program
        Parametric Quil Program with the annealing circuit.

    """
    (reg, qubits_singles, qubits_pairs, n_steps) =\
        (params.reg, params.qubits_singles,
         params.qubits_pairs, params.n_steps)

    p = Program()
    # create list of memory references to store angles in.
    # Has to be so nasty, because aliased memories are not supported yet.
    # Also length 0 memory references crash the QVM
    betas = []
    gammas_singles = []
    gammas_pairs = []
    for i in range(n_steps):
        beta = p.declare('x_rotation_angles{}'.format(i),
                         memory_type='REAL',
                         memory_size=len(reg))
        betas.append(beta)
        if not reg:  # remove length 0 references again
            p.pop()

        gamma_singles = p.declare('z_rotation_angles{}'.format(i),
                                  memory_type='REAL',
                                  memory_size=len(qubits_singles))
        gammas_singles.append(gamma_singles)
        if not qubits_singles:   # remove length 0 references again
            p.pop()

        gamma_pairs = p.declare('zz_rotation_angles{}'.format(i),
                                memory_type='REAL',
                                memory_size=len(qubits_pairs))
        gammas_pairs.append(gamma_pairs)
        if not qubits_pairs:  # remove length 0 references again
            p.pop()

    # apply cost and mixing hamiltonian alternating
    for i in range(n_steps):
        p += _qaoa_cost_ham_rotation(gammas_pairs[i], qubits_pairs,
                                     gammas_singles[i], qubits_singles)
        p += _qaoa_mixing_ham_rotation(betas[i], reg)
    return p


def _all_plus_state(reg: Iterable) -> Program:
    """Prepare the |+>...|+> state on all qubits in reg."""
    p = Program()
    for qubit in reg:
        p.inst(H(qubit))
    return p


def prepare_qaoa_ansatz(initial_state: Program,
                        qaoa_params: Type[AbstractParams]) -> Program:
    """Create parametric quil code for QAOA circuit.

    Parameters
    ----------
    state_prep_program:
        Returns a program for preparation of the initial state
    qaoa_params:
        The parameters of the QAOA circuit.

    Returns
    -------
    Program
        Parametric Quil Program with the whole circuit.

    """
    p = initial_state
    p += _qaoa_annealing_program(qaoa_params)
    return p


def make_qaoa_memory_map(qaoa_params: Type[AbstractParams]) -> dict:
    """Make a memory map for the QAOA Ansatz as produced by
    `prepare_qaoa_ansatz`.

    Parameters
    ----------
    qaoa_params:
        QAOA parameters to take angles from

    Returns
    -------
    Dict:
        A memory_map as expected by QVM.run().

    """
    memory_map = {}
    for i in range(qaoa_params.n_steps):
        memory_map['x_rotation_angles{}'.format(i)] = qaoa_params.x_rotation_angles[i]
        memory_map['z_rotation_angles{}'.format(i)] = qaoa_params.z_rotation_angles[i]
        memory_map['zz_rotation_angles{}'.format(i)] = qaoa_params.zz_rotation_angles[i]
    return memory_map


class QAOACostFunctionOnWFSim(PrepareAndMeasureOnWFSim):
    """
    A cost function that inherits from PrepareAndMeasureOnWFSim and implements
    the specifics of QAOA

    Parameters
    ----------
    hamiltonian:
        The cost hamiltonian
    params:
        Form of the QAOA parameters (with n_steps and type fixed for this instance)
    sim:
        connection to the WavefunctionSimulator to run the simulation on
        Automatically creates one, if None is passed.
    scalar_cost_function:
        If True: __call__ returns only the expectation value
        If False: __call__ returns a tuple (exp_val, std_dev)
        Defaults to True.
    nshots:
        Number of shots to assume to simulate the sampling noise. 0
        corresponds to no sampling noise added and is the default.
    log:
        List to keep log of function calls
    initial_state:
        A Program to run for state preparation. Defaults to
        applying a Hadamard on each qubit (all plust state).
    qubit_mapping:
        A mapping to fix QubitPlaceholders to physical qubits. E.g.
        pyquil.quil.get_default_qubit_mapping(program) gives you on.

    """

    def __init__(self,
                 hamiltonian: PauliSum,
                 params: Type[AbstractParams],
                 sim: WavefunctionSimulator = None,
                 scalar_cost_function: bool = True,
                 nshots: int = 0,
                 enable_logging: bool = False,
                 initial_state: Program = None,
                 qubit_mapping: Dict[QubitPlaceholder,
                                     Union[Qubit, int]] = None):
        """The constructor. See class documentation."""
        if initial_state is None:
            initial_state = _all_plus_state(params.reg)

        self.params = params
        super().__init__(prepare_qaoa_ansatz(initial_state, params),
                         make_memory_map=make_qaoa_memory_map,
                         hamiltonian=hamiltonian,
                         sim=sim,
                         scalar_cost_function=scalar_cost_function,
                         nshots=nshots,
                         enable_logging=enable_logging,
                         qubit_mapping=qubit_mapping)

    def __call__(self, params, nshots: int = None):
        self.params.update_from_raw(params)
        out = super().__call__(self.params, nshots=nshots)

        # remove last entry from the log and replace it with something
        # immutable
        try:
            self.log[-1] = LogEntry(x=deepcopy(params),
                                    fun=self.log[-1].fun)
        except AttributeError:
            pass

        return out

    def get_wavefunction(self,
                         params: Union[list, np.ndarray]) -> Wavefunction:
        """Same as __call__ but returns the wavefunction instead of cost

        Parameters
        ----------
        params:
            _Raw_(!) QAOA parameters for the state preparation. Can be obtained
            from Type[AbstractParams] objects via ``.raw()``

        Returns
        -------
        Wavefunction
            The wavefunction prepared with raw QAOA parameters ``params``
        """
        self.params.update_from_raw(params)
        return super().get_wavefunction(self.params)


class QAOACostFunctionOnQVM(PrepareAndMeasureOnQVM):
    """
    A cost function that inherits from PrepareAndMeasureOnQVM and implements
    the specifics of QAOA

    Parameters
    ----------
    hamiltonian:
        The cost hamiltonian
    params:
        Form of the QAOA parameters (with n_steps and type fixed for this
        instance)
    qvm:
        Connection the QC to run the program on OR a name string like expected
        by ``pyquil.api.get_qc``
    scalar_cost_function:
        If True: __call__ returns only the expectation value
        If False: __call__ returns a tuple (exp_val, std_dev)
        Defaults to True.
    nshots:
        Fixed multiple of ``base_numshots`` for each estimation of the
        expectation value. Defaults to 1
    base_numshots:
        numshots multiplier to compile into the binary. The argument nshots of
         __call__ is then a multplier of this.
    log:
        List to keep log of function calls
    qubit_mapping:
        A mapping to fix QubitPlaceholders to physical qubits. E.g.
        pyquil.quil.get_default_qubit_mapping(program) gives you on.
     """

    def __init__(self,
                 hamiltonian: PauliSum,
                 params: Type[AbstractParams],
                 qvm: Union[QuantumComputer, str],
                 scalar_cost_function: bool = True,
                 nshots: int = 1,
                 base_numshots: int = 100,
                 enable_logging: bool = False,
                 initial_state: Program = None,
                 qubit_mapping: Dict[QubitPlaceholder,
                                     Union[Qubit, int]] = None):
        """The constructor. See class documentation for details."""
        if initial_state is None:
            initial_state = _all_plus_state(params.reg)

        self.params = params
        super().__init__(prepare_qaoa_ansatz(initial_state, params),
                         make_memory_map=make_qaoa_memory_map,
                         hamiltonian=hamiltonian,
                         qvm=qvm,
                         scalar_cost_function=scalar_cost_function,
                         nshots=nshots,
                         base_numshots=base_numshots,
                         enable_logging=enable_logging,
                         qubit_mapping=qubit_mapping)

    def __call__(self, params, nshots: int = None):
        self.params.update_from_raw(params)
        out = super().__call__(self.params, nshots=nshots)
        # remove last entry from the log and replace it with something
        # immutable
        try:
            self.log[-1] = LogEntry(x=deepcopy(params),
                                    fun=self.log[-1].fun)
        except AttributeError:
            pass
        return out
