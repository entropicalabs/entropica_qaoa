"""
Implementation of the QAOA cost_functions. We inherit from vqe/cost_functions
and change only the QAOA specific details.

TODO
----
Change type of `reg` to Iterable or create custom type for it.
"""


from typing import Union, List, Type, Dict

from pyquil import Program
from pyquil.quil import MemoryReference, QubitPlaceholder, Qubit
from pyquil.gates import RX, RZ, CPHASE, H, X, I
from pyquil.paulis import PauliSum
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.api._quantum_computer import QuantumComputer

from forest_qaoa.vqe.cost_function import PrepareAndMeasureOnQVM, PrepareAndMeasureOnWFSim
from forest_qaoa.qaoa.parameters import AbstractQAOAParameters, GeneralQAOAParameters


def _qaoa_mixing_ham_rotation(betas: MemoryReference,
                              reg: Union[List, range]) -> Program:
    """Produce parametric Quil-Code for the mixing hamiltonian rotation.

    Parameters
    ----------
    betas : MemoryReference
        Classic register to read the x_rotation_angles from.
    reg : Union[List, Range]
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
    gammas_pairs : MemoryReference
        Classic register to read the zz_rotation_angles from.
    qubit_pairs : List
        List of the Qubit pairs to apply rotations on.
    gammas_singles : MemoryReference
        Classic register to read the z_rotation_angles from.
    qubit_singles : List
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


# TODO check, whether aliased angles are supported yet
def _qaoa_annealing_program(qaoa_params: Type[AbstractQAOAParameters]) -> Program:
    """Create parametric quil code for QAOA annealing circuit.

    Parameters
    ----------
    qaoa_params : Type[AbstractQAOAParameters]
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


def _prepare_all_plus_state(reg) -> Program:
    """Prepare the |+>...|+> state on all qubits in reg."""
    p = Program()
    for qubit in reg:
        p.inst(H(qubit))
    return p

def _prepare_custom_classical_state(reg, state) -> Program:
    """Prepare a custom classical state for all qubits in reg.
     Parameters
    ----------
    state : Type[list]
        A list of 0s and 1s which represent the starting state of the register, bit-wise.

    Returns
    -------
    Program
        Parametric Quil Program with a circuit in an initial classical state.    
    """
    
    if len(reg) != len(state):
        raise ValueError("qubit state must be the same length as reg")
    
    p = Program()
    for qubit, s in zip(reg, state):
    # if int(s) == 0 we don't need to add any gates, since the qubit is in state 0 by default
        if int(s) == 1:
            p.inst(X(qubit))
    return p


def prepare_qaoa_ansatz(qaoa_params: Type[AbstractQAOAParameters], init_state=None) -> Program:
    """Create parametric quil code for QAOA circuit.

    Parameters
    ----------
    qaoa_params : Type[AbstractQAOAParameters]
        The parameters of the QAOA circuit.
    init_state : Type[list<int>]
        A list of 0s and 1s which represent the starting state of the QAOA circuit, bit-wise.

    Returns
    -------
    Program
        Parametric Quil Program with the whole circuit.

    """
    if not init_state:
        p = _prepare_all_plus_state(qaoa_params.reg)
    else: 
        p = _prepare_custom_classical_state(qaoa_params.reg, init_state)
    p += _qaoa_annealing_program(qaoa_params)
    return p


def make_qaoa_memory_map(qaoa_params: Type[AbstractQAOAParameters]) -> dict:
    """Make a memory map for the QAOA Ansatz as produced by `prepare_qaoa_ansatz`.

    Parameters
    ----------
    qaoa_params : Type(AbstractQAOAParameters)
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
    """

    def __init__(self,
                 hamiltonian: PauliSum,
                 params: Type[AbstractQAOAParameters],
                 sim: WavefunctionSimulator,
                 return_standard_deviation=False,
                 noisy=False,
                 log=None,
                 init_state=None,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None):
        """Create a cost-function for QAOA.

        Parameters
        ----------
        hamiltonian : PauliSum
            The cost hamiltonian
        params : Type[AbstractQAOAParameters]
            Form of the QAOA parameters (with timesteps and type fixed for this instance)
        sim : WavefunctionSimulator
            connection to the WavefunctionSimulator to run the simulation on
        return_standard_deviation : bool
            return standard deviation or only expectation value?
        noisy : False
            Add simulated samplign noise?
        log : list
            List to keep log of function calls
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]
            A mapping to fix QubitPlaceholders to physical qubits. E.g.
            pyquil.quil.get_default_qubit_mapping(program) gives you on.

        """
        self.params = params
        self.init_state = init_state
        
        super().__init__(prepare_qaoa_ansatz(params,init_state=init_state),
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

    def get_wavefunction(self, params):
        """Same as __call__ but returns the wavefunction instead of cost

        Parameters
        ----------
        params: Union[list, np.ndarray]
            Raw(!) QAOA parameters for the state preparation. Can be obtained
            from Type[AbstractQAOAParameters] objects via ``.raw()``

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
    """

    def __init__(self,
                 hamiltonian: PauliSum,
                 params: Type[AbstractQAOAParameters],
                 qvm: QuantumComputer,
                 return_standard_deviation=False,
                 base_numshots: int = 100,
                 log=None,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None):
        """Create a cost-function for QAOA.

        Parameters
        ----------
        hamiltonian : PauliSum
            The cost hamiltonian
        params : Type[AbstractQAOAParameters]
            Form of the QAOA parameters (with timesteps and type fixed for this instance)
        qvm : QuantumComputer
            connection to the QuantumComputer to run on
        return_standard_deviation : bool
            return standard deviation or only expectation value?
        param base_numshots : int
            numshots to compile into the binary. The argument nshots of __call__
            is then a multplier of this.
        log : list
            List to keep log of function calls
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]
            A mapping to fix QubitPlaceholders to physical qubits. E.g.
            pyquil.quil.get_default_qubit_mapping(program) gives you on.


        """
        self.params = params
        super().__init__(prepare_qaoa_ansatz(params),
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
