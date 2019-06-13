"""
Implementation of the QAOA cost functions. We inherit from
``vqe.cost_functions`` and change only the QAOA specific details.
"""


from typing import Union, List, Type, Dict, Iterable, Callable
import numpy as np

from pyquil import Program
from pyquil.quil import MemoryReference, QubitPlaceholder, Qubit
from pyquil.wavefunction import Wavefunction
from pyquil.gates import RX, RZ, CPHASE, H
from pyquil.paulis import PauliSum
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.api._quantum_computer import QuantumComputer

from vqe.cost_function import PrepareAndMeasureOnQVM, PrepareAndMeasureOnWFSim
from qaoa.parameters import AbstractQAOAParameters, GeneralQAOAParameters


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


def _qaoa_annealing_program(qaoa_params: Type[AbstractQAOAParameters]) -> Program:
    """Create parametric quil code for the QAOA annealing circuit.

    Parameters
    ----------
    qaoa_params:
        The parameters of the QAOA circuit.

    Returns
    -------
    Program
        Parametric Quil Program with the annealing circuit.

    """
    (reg, qubits_singles, qubits_pairs, timesteps) =\
        (qaoa_params.reg, qaoa_params.qubits_singles,
         qaoa_params.qubits_pairs, qaoa_params.timesteps)

    p = Program()
    # create list of memory references to store angles in.
    # Has to be so nasty, because aliased memories are not supported yet.
    # Also length 0 memory references crash the QVM
    betas = []
    gammas_singles = []
    gammas_pairs = []
    for i in range(timesteps):
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
    for i in range(timesteps):
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
                        qaoa_params: Type[AbstractQAOAParameters]) -> Program:
    """Create parametric quil code for QAOA circuit.

    Parameters
    ----------
    initial_state:
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


def make_qaoa_memory_map(qaoa_params: Type[AbstractQAOAParameters]) -> dict:
    """Make a memory map for the QAOA Ansatz as produced by `prepare_qaoa_ansatz`.

    Parameters
    ----------
    qaoa_params:
        QAOA parameters to take angles from

    Returns
    -------
    dict:
        A memory_map as expected by QVM.run().

    """
    memory_map = {}
    for i in range(qaoa_params.timesteps):
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
        Form of the QAOA parameters (with timesteps and type fixed for this instance)
    sim:
        connection to the WavefunctionSimulator to run the simulation on
    return_standard_deviation:
        return standard deviation or only expectation value?
    noisy:
        Add simulated sampling noise?
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
                 params: Type[AbstractQAOAParameters],
                 sim: WavefunctionSimulator,
                 return_standard_deviation: bool =False,
                 noisy: bool =False,
                 log: List =None,
                 initial_state: Program = None,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None):
        """The constructor. See class documentation."""
        if initial_state is None:
            initial_state = _all_plus_state(params.reg)

        self.params = params
        super().__init__(prepare_qaoa_ansatz(initial_state, params),
                         make_memory_map=make_qaoa_memory_map,
                         hamiltonian=hamiltonian,
                         sim=sim,
                         return_standard_deviation=return_standard_deviation,
                         noisy=noisy,
                         log=log,
                         qubit_mapping=qubit_mapping)

    def __call__(self, params, nshots: int = 1000):
        self.params.update_from_raw(params)
        out = super().__call__(self.params, nshots=nshots)
        return out

    def get_wavefunction(self, params: Union[list, np.array]) -> Wavefunction:
        """Same as ``__call__`` but returns the wavefunction instead of cost

        Parameters
        ----------
        params:
            _Raw_(!) QAOA parameters for the state preparation. Can be obtained
            from Type[AbstractQAOAParameters] objects via ``qaoa_params.raw()``

        Returns
        -------
        Wavefunction
            The wavefunction prepared with raw QAOA parameters ``qaoa_params``
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
        Form of the QAOA parameters (with timesteps and type fixed for this instance)
    qvm:
        connection to the QuantumComputer to run on
    return_standard_deviation:
        return standard deviation or only expectation value?
    param base_numshots:
        numshots to compile into the binary. The argument nshots of __call__
        is then a multplier of this.
    log:
        List to keep log of function calls
    qubit_mapping:
        A mapping to fix QubitPlaceholders to physical qubits. E.g.
        pyquil.quil.get_default_qubit_mapping(program) gives you on.
    """

    def __init__(self,
                 hamiltonian: PauliSum,
                 params: Type[AbstractQAOAParameters],
                 qvm: QuantumComputer,
                 return_standard_deviation: bool =False,
                 base_numshots: int = 100,
                 log: list =None,
                 initial_state: Program = None,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None):
        """The constructor. See class documentation for details"""
        if initial_state is None:
            initial_state = _all_plus_state(params.reg)

        self.params = params
        super().__init__(prepare_qaoa_ansatz(initial_state, params),
                         make_memory_map=make_qaoa_memory_map,
                         hamiltonian=hamiltonian,
                         qvm=qvm,
                         return_standard_deviation=return_standard_deviation,
                         base_numshots=base_numshots,
                         log=log,
                         qubit_mapping=qubit_mapping)

    def __call__(self, params, nshots: int = 10):
        self.params.update_from_raw(params)
        out = super().__call__(self.params, nshots=nshots)
        return out
