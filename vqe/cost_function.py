"""
Different cost functions for VQE and one abstract template.
"""
from typing import Callable, Iterable
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program
from pyquil.api._wavefunction_simulator import WavefunctionSimulator
from pyquil.api._quantum_computer import QuantumComputer


import numpy as np


class abstract_cost_function():
    """
    Template class for cost_functions that are passed to the optimizer.
    """

    def __init__(return_float: bool = False, log=None):
        """Set up the cost function.

        Parameters
        ----------
        return_float : bool
            Return the cost as a float for scalar optimizers or as a tuple
            (cost, sigma_cost) for optimizers of noisy functions.
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


class prep_and_measure_ham_qvm(abstract_cost_function):
    """A cost function that prepares an ansatz and measures its energy w.r.t
       hamiltonian on the qvm
    """

    def __init__(self,
                 prepare_ansatz: Callable[[Iterable], Program],
                 hamiltonian: PauliSum,
                 sim: WavefunctionSimulator,
                 return_float=True,
                 noisy=False,
                 log=None):
        """Set up the cost_function.

        Parameters
        ----------
        prepare_ansatz : function
            A function prepare_ansatz(params) -> pyquil.quil.Program
            that creates a pyquil program to prepare the state with parameters
            params.
        hamiltonian : PauliSum
            The hamiltonian w.r.t which to measure the energy.
        sim : WavefunctionSimulator
            A WavefunctionSimulator instance to get the wavefunction from.
        return_float : bool
            Return the cost as a float for scalar optimizers or as a tuple
            (cost, sigma_cost) for optimizers of noisy functions.
            (the default is False).
        noisy: bool
            Add simulated noise to the energy? (the default is False)
        log : list
            A list to write a log of function values to. If None is passed no
            log is created.
        """
        self.prepare_ansatz = prepare_ansatz
        self.return_float = return_float
        self.noisy = noisy
        self.sim = sim  # TODO start own simulator, if None is passed

        # TODO What if prepare_ansatz acts on more qubits than ham?
        # then hamiltonian and wavefunction don't fit together...
        if isinstance(hamiltonian, PauliSum):
            nqubits = max(hamiltonian.get_qubits()) + 1
            self.ham = hamiltonian.matrix(nqubits=nqubits)
        elif isinstance(hamiltonian, (numpy.matrix, numpy.ndarray)):
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
        params : Iterable
            Parameters of the state preparation circuit.
        nshots : int
            Number of shots to take to estimate the energy (the default is 1000).

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.
        """
        wf = self.sim.wavefunction(self.prepare_ansatz(params))
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

        if self.return_float:
            return out[0]

        return out

# TODO fix this
class prep_and_measure_ham_qc(abstract_cost_function):
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
                 return_float: bool = True,
                 base_numshots: int = 100,
                 log: list = None):
        """
        Parameters
        ----------
        param prepare_ansatz: Program
            A parametric pyquil program for the state preparation
        param make_memory_map: Function
            A function that creates a memory map from the array of parameters
        param hamiltonian : PauliSum
            The hamiltonian
        param qvm : Quantum Computer connection
            Connection the QC to run the program on.
        param return_float : bool
            return a float or tuple of energy and its standard deviation.
        param base_numshots : int
            numshots to compile into the binary. The argument nshots of __call__
            is then a multplier of this.
        """
        # TODO sanitize input?
        self.qc = qc
        self.ham = ham
        append_measure_register(prepare_ansatz, params.reg, trials=base_numshots)
        self.exe = qc.compile(prepare_ansatz)

        if log is not None:
            self.log = log

    def __call__(self, params, N=1):
        """
        Parameters
        ----------
        :param params:    (1D array)   raw qaoa_parameters
        :param N:         (int)        Number of times to run exe
        """
        self.params.update(params)
        memory_map = make_memory_map(self.params)

        bitstrings = self.qc.run(self.exe, memory_map=memory_map)
        for i in range(N - 1):
            bitstrings = np.append(bitstrings, self.qc.run(self.exe, memory_map=memory_map), axis=0)

        res = energy_expecation_value(self.ham, bitstrings)
        try:
            self.log.append(res)
        except AttributeError:
            pass
        return res
