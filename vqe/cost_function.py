"""
Different cost functions for VQE and one abstract template.
"""
from typing import Callable, Iterable, Union, List, Dict

from vqe.measurelib import append_measure_register, hamiltonian_expectation_value

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, Qubit, QubitPlaceholder, address_qubits
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.api._quantum_computer import QuantumComputer

import numpy as np


class AbstractCostFunction():
    """
    Template class for cost_functions that are passed to the optimizer.
    """

    def __init__(return_standard_deviation: bool = False, log=None):
        """Set up the cost function.

        Parameters
        ----------
        return_standard_deviation : bool
            Return the cost as a float for scalar optimizers or as a tuple
            (cost, cost_standard_deviation) for optimizers of noisy functions.
            (the default is False).
        log : list
            A list to write a log of function values to. If None is passed no
            log is created.
        """
        raise NotImplementedError()

    def __call__(params, nshots: int):
        """Estimate cost_functions(params) with nshots samples

        Parameters
        ----------
        params : ndarray, shape (n,)
            Parameters of the state preparation circuit. Array of size `n` where
            `n` is the number of different parameters.
        nshots : int
            Number of shots to take to estimate `cost_function(params)`

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.

        """
        raise NotImplementedError()


# TODO support hamiltonians with qubit QubitPlaceholders?
# TODO Join code with PrepareAndMeasureOnQVM, since they are _very_ similar now?
class PrepareAndMeasureOnWFSim(AbstractCostFunction):
    """A cost function that prepares an ansatz and measures its energy w.r.t
       hamiltonian on the qvm
    """

    def __init__(self,
                 prepare_ansatz: Program,
                 make_memory_map: Callable[[Union[List, np.array, np.matrix]], Dict],
                 hamiltonian: Union[PauliSum, np.array],
                 sim: WavefunctionSimulator,
                 return_standard_deviation=False,
                 noisy=False,
                 log=None,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None):
        """Set up the cost_function.

        Parameters
        ----------
        param prepare_ansatz: Program
            A parametric pyquil program for the state preparation
        param make_memory_map: Function
            A function that creates a memory map from the array of parameters
        hamiltonian : PauliSum
            The hamiltonian w.r.t which to measure the energy.
        sim : WavefunctionSimulator
            A WavefunctionSimulator instance to get the wavefunction from.
        return_standard_deviation : bool
            Return the cost as a float for scalar optimizers or as a tuple
            (cost, cost_standard_deviation) for optimizers of noisy functions.
            (the default is False).
        noisy: bool
            Add simulated noise to the energy? (the default is False)
        log : list
            A list to write a log of function values to. If None is passed no
            log is created.
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]
            A mapping to fix QubitPlaceholders to physical qubits. E.g.
            pyquil.quil.get_default_qubit_mapping(program) gives you on.
        """
        self.make_memory_map = make_memory_map
        self.return_standard_deviation = return_standard_deviation
        self.noisy = noisy
        self.sim = sim  # TODO start own simulator, if None is passed

        # TODO automatically generate Qubit mapping, if None is passed?
        # TODO ask Rigetti to implement "<" between qubits?
        if qubit_mapping is not None:
            int_mapping = dict(zip(qubit_mapping.keys(),
                                   [q.index for q in qubit_mapping.values()]))
            self.prepare_ansatz = address_qubits(prepare_ansatz, qubit_mapping)
        else:
            int_mapping = None
            self.prepare_ansatz = prepare_ansatz

        # TODO What if prepare_ansatz acts on more qubits than ham?
        # then hamiltonian and wavefunction don't fit together...
        if isinstance(hamiltonian, PauliSum):
            nqubits = len(hamiltonian.get_qubits())
            self.ham = hamiltonian.matrix(int_mapping or {}, nqubits)
        elif isinstance(hamiltonian, (np.matrix, np.ndarray)):
            self.ham = hamiltonian
        else:
            raise ValueError(
                "hamiltonian has to be a PauliSum or numpy matrix")

        self.ham_squared = self.ham**2

        if log is not None:
            self.log = log

    def __call__(self, params, nshots: int = 1000):
        """Cost function that computes <psi|ham|psi> with psi prepared with
        prepare_ansatz(params).

        Parameters
        ----------
        params : Union[list, np.ndarray]
            Parameters of the state preparation circuit.
        nshots : int
            Number of shots to take to estimate the energy (the default is 1000).

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.
        """
        memory_map = self.make_memory_map(params)
        wf = self.sim.wavefunction(self.prepare_ansatz, memory_map=memory_map)
        wf = np.reshape(wf.amplitudes, (-1, 1))
        E = np.conj(wf).T.dot(self.ham.dot(wf)).real
        sigma_E = nshots**(-1 / 2) * (
            np.conj(wf).T.dot(self.ham_squared.dot(wf)).real - E**2)

        # add simulated noise, if wanted
        if self.noisy:
            E += np.random.randn()*sigma_E
        out = (float(E), float(sigma_E))

        try:
            self.log.append(out)
        except AttributeError:
            pass

        if  not self.return_standard_deviation:
            return out[0]
        else:
            return out


# TODO fix this
class PrepareAndMeasureOnQVM(AbstractCostFunction):
    """A cost function that prepares an ansatz and measures its energy w.r.t
       hamiltonian on a quantum computer (or simulator).

       This cost_function makes use of pyquils parametric circuits and thus
       has to be supplied with a parametric circuit and a function to create
       memory maps that can be passed to qvm.run.
    """

    def __init__(self,
                 prepare_ansatz: Program,
                 make_memory_map: Callable[[Iterable],dict],
                 hamiltonian: PauliSum,
                 qvm: QuantumComputer,
                 return_standard_deviation: bool = False,
                 base_numshots: int = 100,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None,
                 log: list = None):
        """
        Parameters
        ----------
        prepare_ansatz: Program
            A parametric pyquil program for the state preparation
        make_memory_map: Function
            A function that creates a memory map from the array of parameters
        hamiltonian : PauliSum
            The hamiltonian
        qvm : Quantum Computer connection
            Connection the QC to run the program on.
        return_standard_deviation : bool
            return a float or tuple of energy and its standard deviation.
        base_numshots : int
            numshots to compile into the binary. The argument nshots of __call__
            is then a multplier of this.
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]
            A mapping to fix all QubitPlaceholders to physical qubits. E.g.
            pyquil.quil.get_default_qubit_mapping(program) gives you on.
        """
        # TODO sanitize input?
        self.qvm = qvm
        self.return_standard_deviation = return_standard_deviation
        self.make_memory_map = make_memory_map
        
        if qubit_mapping is not None:
            prepare_ansatz = address_qubits(prepare_ansatz, qubit_mapping)
            self.ham = address_qubits_hamiltonian(hamiltonian, qubit_mapping)
        else:
            self.ham = hamiltonian

        if log is not None:
            self.log = log

        append_measure_register(prepare_ansatz, qubits=self.ham.get_qubits(), trials=base_numshots)
        self.exe = qvm.compile(prepare_ansatz)


    def __call__(self, params, nshots=1):
        """
        Parameters
        ----------
        param params :  1D array
            the parameters to run the state preparation circuit with
        param N : int
            Number of times to run exe
        """
        memory_map = self.make_memory_map(params)

        bitstrings = self.qvm.run(self.exe, memory_map=memory_map)
        for i in range(nshots - 1):
            bitstrings = np.append(bitstrings, self.qvm.run(self.exe, memory_map=memory_map), axis=0)

        res = hamiltonian_expectation_value(self.ham, bitstrings)
        try:
            self.log.append(res)
        except AttributeError:
            pass

        if not self.return_standard_deviation:
            return res[0]
        else:
            return res


def address_qubits_hamiltonian(hamiltonian: PauliSum,
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]) -> PauliSum:
    """Map Qubit Placeholders to ints in a PauliSum.

    Parameters
    ----------
    hamiltonian : PauliSum
        The PauliSum.
    qubit_mapping : Dict[QubitPlaceholder, Union[Qubit, int]]
        A qubit_mapping. e.g. provided by pyquil.quil.get_default_qubit_mapping.

    Returns
    -------
    PauliSum
        A PauliSum with remapped Qubits.

    Note
    ----
    This code relies completely on going all the way down the rabbits hole
    with for loops. It would be preferable to have this functionality in
    pyquil.paulis.PauliSum directly
    """
    out = PauliTerm("I", 0, 0)
    # Make sure we map to integers and not to Qubits(), these are not
    # supported by pyquil.paulis.PauliSum().
    if set([Qubit]) == set(map(type, qubit_mapping.values())):
        qubit_mapping = dict(zip(qubit_mapping.keys(),
                                 [q.index for q in qubit_mapping.values()]))
    # And change all of them
    for term in hamiltonian:
        coeff = term.coefficient
        ops = []
        for factor in term:
            ops.append((factor[1], qubit_mapping[factor[0]]))
        out += PauliTerm.from_list(ops, coeff)
    return out
