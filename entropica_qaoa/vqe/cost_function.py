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
Different cost functions for VQE and one abstract template.
"""
from typing import Callable, Iterable, Union, List, Dict, Tuple
import numpy as np
from collections import namedtuple
from copy import deepcopy
import warnings

from pyquil.paulis import PauliSum, PauliTerm
from pyquil.quil import Program, Qubit, QubitPlaceholder, address_qubits
from pyquil.wavefunction import Wavefunction
from pyquil.api import WavefunctionSimulator, QuantumComputer, get_qc

from entropica_qaoa.vqe.measurelib import (append_measure_register,
                                           commuting_decomposition,
                                           sampling_expectation,
                                           kron_eigs,
                                           base_change_fun,
                                           wavefunction_expectation)


LogEntry = namedtuple("LogEntry", ['x', 'fun'])


class AbstractCostFunction():
    """Template class for cost_functions that are passed to the optimizer

    Parameters
    ----------
    scalar_cost_function:
        If ``True``: self.__call__ has  signature
        ``(x, nshots) -> (exp_val, std_val)``
        If ``False``: ``self.__call__()`` has  signature ``(x) -> (exp_val)``,
        but the ``nshots`` argument in ``__init__`` has to be given.
    nshots:
        Optional.  Has to be given, if ``scalar_cost_function``
        is ``False``
        Number of shots to take for cost function evaluation.
    enable_logging:
        If true, a log is created which contains the parameter and function
        values at each function call. It is a list of namedtuples of the form
        ("x", "fun")

    """

    def __init__(self,
                 scalar_cost_function: bool = True,
                 nshots: int = 0,
                 enable_logging: bool = False):
        """The constructor. See class docstring"""
        raise NotImplementedError()

    def __call__(self,
                 params: np.array,
                 nshots: int = None) -> Union[float, tuple]:
        """Estimate cost_functions(params) with nshots samples

        Parameters
        ----------
        params :
            Parameters of the state preparation circuit. Array of size `n`
            where `n` is the number of different parameters.
        nshots:
            Number of shots to take to estimate ``cost_function(params)``
            Overrides whatever was passed in ``__init__``

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.

        """
        raise NotImplementedError()


# TODO support hamiltonians with qubit QubitPlaceholders?
class PrepareAndMeasureOnWFSim(AbstractCostFunction):
    """A cost function that prepares an ansatz and measures its energy w.r.t
    ``hamiltonian`` on the qvm

    Parameters
    ----------
    param prepare_ansatz:
        A parametric pyquil program for the state preparation
    param make_memory_map:
        A function that creates a memory map from the array of parameters
    hamiltonian:
        The hamiltonian with respect to which to measure the energy.
    sim:
        A WavefunctionSimulator instance to get the wavefunction from.
        Automatically created, if None is passed.
    scalar_cost_function:
        If True: __call__ returns only the expectation value
        If False: __call__ returns a tuple (exp_val, std_dev)
        Defaults to True.
    nshots:
        Number of shots to assume to simulate the sampling noise. 0
        corresponds to no sampling noise added and is the default.
    enable_logging:
        If true, a log is created which contains the parameter and function
        values at each function call. It is a list of namedtuples of the form
        ("x", "fun")
    qubit_mapping:
        A mapping to fix QubitPlaceholders to physical qubits. E.g.
        pyquil.quil.get_default_qubit_mapping(program) gives you on.
    hamiltonian_is_diagonal:
        Pass true, if the hamiltonian contains only Z terms and products
        thereof. This speeds up __init__ considerably. Defaults to False.
    """

    def __init__(self,
                 prepare_ansatz: Program,
                 make_memory_map: Callable[[np.array], Dict],
                 hamiltonian: PauliSum,
                 sim: WavefunctionSimulator = None,
                 scalar_cost_function: bool = True,
                 nshots: int = 0,
                 enable_logging: bool = False,
                 qubit_mapping: Dict[QubitPlaceholder,
                                     Union[Qubit, int]] = None,
                 hamiltonian_is_diagonal: bool =False):

        self.scalar = scalar_cost_function
        self.nshots = nshots
        self.make_memory_map = make_memory_map

        if sim is None:
            sim = WavefunctionSimulator()
        self.sim = sim

        # TODO automatically generate Qubit mapping, if None is passed?
        # TODO ask Rigetti to implement "<" between qubits?
        if qubit_mapping is not None:
            self.prepare_ansatz = address_qubits(prepare_ansatz, qubit_mapping)
            ham = address_qubits_hamiltonian(hamiltonian, qubit_mapping)
        else:
            self.prepare_ansatz = prepare_ansatz
            ham = hamiltonian

        qubits = list(self.prepare_ansatz.get_qubits())
        with warnings.catch_warnings():   # supress commutation warnings
            warnings.simplefilter("ignore")
            ham_squared = ham * ham
        # decompose the hamiltonian in simultaneously measurable parts
        if not hamiltonian_is_diagonal:
            hams = commuting_decomposition(ham)
            hams_squared = commuting_decomposition(ham_squared)
        else:
            hams = [ham]
            hams_squared = [ham_squared]
        # get eigenvalues of each term in the decompsition and base changes
        self.hams_eigs = [kron_eigs(ham, qubits) for ham in hams]
        self.hams_squared_eigs = [kron_eigs(ham, qubits)
                                  for ham in hams_squared]
        self.base_changes = [base_change_fun(ham, qubits) for ham in hams]
        self.base_changes_squared = [base_change_fun(ham, qubits)
                                     for ham in hams_squared]

        # prepare logging if wished
        if enable_logging:
            self.log = []

    def __call__(self,
                 params: Union[list, np.ndarray],
                 nshots: int = None) -> Union[float, Tuple]:
        """Cost function that computes <psi|ham|psi> with psi prepared with
        prepare_ansatz(params).

        Parameters
        ----------
        params:
            Parameters of the state preparation circuit.
        nshots:
            Overrides nshots from __init__ if passed

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.
        """
        if nshots is None:
            nshots = self.nshots

        memory_map = self.make_memory_map(params)
        wf = self.sim.wavefunction(self.prepare_ansatz, memory_map)
        E, E2 = wavefunction_expectation(self.hams_eigs, self.base_changes,
                                         self.hams_squared_eigs,
                                         self.base_changes_squared,
                                         wf.amplitudes)

        if nshots:
            sigma_E = np.sqrt(
                (E2 - E**2).real / nshots)
        else:
            sigma_E = 0

        E += np.random.randn() * sigma_E
        out = (E, sigma_E)  # Todo: Why the float casting?

        # Append function value and params to the log.
        # deepcopy is needed, because x may be a mutable type.
        try:
            self.log.append(LogEntry(x=deepcopy(params),
                                     fun=out))
        except AttributeError:
            pass

        # and return the expectation value or (exp_val, std_dev)
        if self.scalar:
            return float(E)
        return out

    def get_wavefunction(self,
                         params: Union[List, np.ndarray]) -> Wavefunction:
        """Same as __call__ but returns the wavefunction instead of cost

        Parameters
        ----------
        params:
            Parameters of the state preparation circuit

        Returns
        -------
        Wavefunction:
            The wavefunction prepared with parameters ``params``
        """
        memory_map = self.make_memory_map(params)
        wf = self.sim.wavefunction(self.prepare_ansatz, memory_map=memory_map)
        return wf


class PrepareAndMeasureOnQVM(AbstractCostFunction):
    """A cost function that prepares an ansatz and measures its energy w.r.t
    hamiltonian on a quantum computer (or simulator).

    This cost_function makes use of pyquils parametric circuits and thus
    has to be supplied with a parametric circuit and a function to create
    memory maps that can be passed to qvm.run.

    Parameters
    ----------
    prepare_ansatz:
        A parametric pyquil program for the state preparation
    make_memory_map:
        A function that creates a memory map from the array of parameters
    hamiltonian:
        The hamiltonian
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
    qubit_mapping:
        A mapping to fix all QubitPlaceholders to physical qubits. E.g.
        pyquil.quil.get_default_qubit_mapping(program) gives you on.
    enable_logging:
        If true, a log is created which contains the parameter and function
        values at each function call. It is a list of namedtuples of the form
        ("x", "fun")
    """

    def __init__(self,
                 prepare_ansatz: Program,
                 make_memory_map: Callable[[Iterable], dict],
                 hamiltonian: PauliSum,
                 qvm: Union[QuantumComputer, str],
                 scalar_cost_function: bool = True,
                 nshots: int = 1,
                 base_numshots: int = 100,
                 qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]] = None,
                 enable_logging: bool = False,
                 hamiltonian_is_diagonal: bool =False):

        self.scalar = scalar_cost_function
        self.nshots = nshots
        self.make_memory_map = make_memory_map

        if isinstance(qvm, str):
            qvm = get_qc(qvm)
        self.qvm = qvm

        if qubit_mapping is not None:
            prepare_ansatz = address_qubits(prepare_ansatz, qubit_mapping)
            ham = address_qubits_hamiltonian(hamiltonian, qubit_mapping)
        else:
            ham = hamiltonian

        if not hamiltonian_is_diagonal:
            self.hams = commuting_decomposition(ham)
        else:
            self.hams = [ham]

        self.exes = []
        for ham in self.hams:
            # need a different program for each of the self commuting hams
            p = prepare_ansatz.copy()
            append_measure_register(p,
                                    qubits=ham.get_qubits(),
                                    trials=base_numshots,
                                    ham=ham)
            self.exes.append(qvm.compile(p))

        if enable_logging:
            self.log = []

    def __call__(self,
                 params: np.array,
                 nshots: int = None) -> Union[float, Tuple]:
        """
        Parameters
        ----------
        param params:
            the parameters to run the state preparation circuit with
        param nshots:
            Overrides nshots in __init__ if passed

        Returns
        -------
        float or tuple (cost, cost_stdev)
            Either only the cost or a tuple of the cost and the standard
            deviation estimate based on the samples.
        """
        if nshots is None:
            nshots = self.nshots

        memory_map = self.make_memory_map(params)

        bitstrings = []
        for exe in self.exes:
            bitstring = self.qvm.run(exe, memory_map=memory_map)
            for i in range(nshots - 1):
                new_bits = self.qvm.run(exe, memory_map=memory_map)
                bitstring = np.append(bitstring, new_bits, axis=0)
            bitstrings.append(bitstring)

        out = sampling_expectation(self.hams, bitstrings)

        # Append function value and params to the log.
        # deepcopy is needed, because x may be a mutable type.
        try:
            self.log.append(LogEntry(x=deepcopy(params),
                                     fun=out))
        except AttributeError:
            pass

        if self.scalar:
            return out[0]
        else:
            return out


def address_qubits_hamiltonian(
        hamiltonian: PauliSum,
        qubit_mapping: Dict[QubitPlaceholder, Union[Qubit, int]]
        ) -> PauliSum:
    """Map Qubit Placeholders to ints in a PauliSum.

    Parameters
    ----------
    hamiltonian:
        The PauliSum.
    qubit_mapping:
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
