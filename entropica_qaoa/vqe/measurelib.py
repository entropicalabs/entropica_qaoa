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
Various convenience functions for measurements on a quantum computer or
wavefunction simulator
"""

from typing import List, Union, Tuple

import numpy as np
import networkx as nx
import itertools

from pyquil.quil import MEASURE, Program, QubitPlaceholder
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import H, I, RX


def append_measure_register(program: Program,
                            qubits: List = None,
                            trials: int = 10,
                            ham: PauliSum = None) -> Program:
    """Creates readout register, MEASURE instructions for register and wraps
    in trials trials.

    Parameters
    ----------
    param qubits : list
        List of Qubits to measure. If None, program.get_qubits() is used
    param trials : int
        The number of trials to run.
    param ham : PauliSum
        Hamiltonian to whose basis we need to switch. All terms in it must
        trivially commute!

    Returns
    -------
    Program :
        program with the gate change and measure instructions appended
    """
    base_change_gates = {'X': lambda qubit: H(qubit),
                         'Y': lambda qubit: RX(np.pi / 2, qubit),
                         'Z': lambda qubit: I(qubit)}

    if qubits is None:
        qubits = program.get_qubits()


    def _get_correct_gate(qubit: Union[int, QubitPlaceholder]) -> Program():
        """Correct base change gate on the qubit `qubit` given `ham`"""
        # this is an extra function, because `return` allows us to
        # easily break out of loops
        for term in ham:
            if term[qubit] != 'I':
                return base_change_gates[term[qubit]](qubit)

        raise ValueError(f"PauliSum {ham} doesn't act on qubit {qubit}")

    # append to correct base change gates if ham is specified. Otherwise
    # assume diagonal basis
    if ham is not None:
        for qubit in ham.get_qubits():
            program += Program(_get_correct_gate(qubit))

    # create a read out register
    ro = program.declare('ro', memory_type='BIT', memory_size=len(qubits))

    # add measure instructions to the specified qubits
    for i, qubit in enumerate(qubits):
        program += MEASURE(qubit, ro[i])
    program.wrap_in_numshots_loop(trials)
    return program


def sampling_expectation_z_base(hamiltonian: PauliSum,
                                bitstrings: np.array) -> Tuple[float, float]:
    """Calculates the energy expectation value of ``bitstrings`` w.r.t ``ham``.

    Warning
    -------
    This function assumes, that all terms in ``hamiltonian`` commute with each
    other _and_ that the ``bitstrings`` were measured in the correct basis

    Parameters
    ----------
    param hamiltonian : PauliSum
        The hamiltonian
    param bitstrings : 2D arry or list
        the measurement outcomes. Columns are outcomes for one qubit.

    Returns
    -------
    tuple (expectation_value, standard_deviation)
    """

    # this dictionary maps from qubit indices to indices of the bitstrings
    # This is neccesary, because hamiltonian might not act on all qubits.
    # E.g. if hamiltonian = X0 + 1.0*Z2 bitstrings is a 2 x numshots array
    index_lut = {q: i for (i, q) in enumerate(hamiltonian.get_qubits())}
    if bitstrings.ndim == 2:
        energies = np.zeros(bitstrings.shape[0])
    else:
        energies = np.array([0])
    for term in hamiltonian:
        sign = np.zeros_like(energies)
        for factor in term:
            sign += bitstrings[:, index_lut[factor[0]]]
        energies += term.coefficient.real * (-1)**sign
    return (np.mean(energies),
            np.sqrt(np.var(energies)) / np.sqrt(bitstrings.shape[0] - 1))


def sampling_expectation(hamiltonians: List[PauliSum],
                         bitstrings: List[np.array]) -> Tuple:
    """Mapped wrapper around ``sampling_expectation_z_base``.

    A function that computes expectation values of a list of hamiltonians
    w.r.t a list of bitstrings. Assumes, that each pair in
    ``zip(hamiltonians, bitstrings)`` is as needed by
    ``sampling_expectation_z_base``

    Parameters
    ----------
    hamiltonians : List[PauliSum]
        List of PauliSums. Each PauliSum must only consist of mutually
        commuting terms
    bitstrings : List[np.array]
        List of the measured bitstrings. Each bitstring must have dimensions
        corresponding to the coresponding PauliSum

    Returns
    -------
    tuple (expectation_value, standard_deviation)
    """
    energies = 0
    var = 0
    for ham, bits in zip(hamiltonians, bitstrings):
        e, s = sampling_expectation_z_base(ham, bits)
        energies += e
        var += s**2

    return (energies, np.sqrt(var))


def _PauliTerms_commute_trivially(term1: PauliTerm, term2: PauliTerm) -> bool:
    """Checks if two PauliTerms commute trivially

    Returns true, if on each qubit both terms have the same sigma matrix
    or if one has only an identity gate
    """
    for factor in term1:
        if factor[1] != term2[factor[0]] and not term2[factor[0]] == 'I':
            return False
    return True


def commuting_decomposition(ham: PauliSum) -> List[PauliSum]:
    """Decompose ham into a list of PauliSums with mutually commuting terms"""

    # create the commutation graph of the hamiltonian
    # connected edges don't commute
    commutation_graph = nx.Graph()
    for term in ham:
        commutation_graph.add_node(term)

    for (term1, term2) in itertools.combinations(ham, 2):
        if not _PauliTerms_commute_trivially(term1, term2):
            commutation_graph.add_edge(term1, term2)

    # color the commutation graph. All terms with one color can be measured
    # simultaneously
    color_map = nx.algorithms.coloring.greedy_color(commutation_graph)

    pauli_sum_list = [False] * (max(color_map.values()) + 1)
    for term in color_map.keys():
        if pauli_sum_list[color_map[term]] is False:
            pauli_sum_list[color_map[term]] = PauliSum([term])
        else:
            pauli_sum_list[color_map[term]] += term

    return pauli_sum_list
