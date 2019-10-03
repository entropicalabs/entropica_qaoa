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
wavefunction simulator.


Some preliminary explanation
============================

Measuring the expectation value of a hamiltonian like

.. math::

    H = a Z_0 Z_1 + b Z_1 Z_2 + c X_0 X_1 + d X_1 + e X_2

requires decomposing the hamiltonian into parts that commute with each other
and then measuring them succesively. On a real QPU we need to do this,
because physics don't allow us to measure non-commuting parts of a
hamiltonian in one run. On the Wavefunction Simulator we need to do this,
because translating :math:`\\left< \\psi | H | \\psi \\right>` naively to
``wf.conj()@ham@wf.conj()`` results in huge and sparse matrices ``ham`` that
take longer to generate, than it takes to calculate the wavefunction ``wf``.

Even though :math:`Z_0 Z_1` and :math:`X_0 X_1` commute (go ahead and check
it), we will measure them in separate runs, because measuring them
simultaenously can't be done by simply measuring the qubits 0 and 1. We only
attempt to measure Pauli products simultaneously that *commute trivially*.
Two Pauli products commute trivially, if on each qubit both act with the same
Pauli Operator, or if either one acts only with the identity. This implies in
particular, that they can be measured without the need for ancilla qubits.
"""

from typing import List, Union, Tuple, Callable, Set
from string import ascii_letters
import warnings

import numpy as np
import networkx as nx
import itertools
import functools

from pyquil.quil import MEASURE, Program, QubitPlaceholder
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.gates import H, I, RX


# ###########################################################################
# Tools to decompose PauliSums into smaller PauliSums that can be measured
# simultaneously
# ###########################################################################


def _PauliTerms_commute_trivially(term1: PauliTerm, term2: PauliTerm) -> bool:
    """Checks if two PauliTerms commute trivially.

    Returns true, if on each qubit both terms have the same Pauli operator
    or if one has only an identity gate
    """
    for factor in term1:
        if factor[1] != term2[factor[0]] and not term2[factor[0]] == 'I':
            return False
    return True


def commuting_decomposition(ham: PauliSum) -> List[PauliSum]:
    """Decompose ham into a list of PauliSums with mutually commuting terms."""

    # create the commutation graph of the hamiltonian
    # connected edges don't commute
    commutation_graph = nx.Graph()
    for term in ham:
        commutation_graph.add_node(term)

    # If the hamiltonian wasn't fully simplified, not all terms get added
    # to the graph. Taking care of this here.
    if len(commutation_graph.nodes) != len(ham):
        warnings.warn(f"The hamiltonian {ham} is not fully simplified. "
                      "Simplifying it now for you.")
        ham.simplify()
        commutation_graph = nx.Graph()
        for term in ham:
            commutation_graph.add_node(term)


    for (term1, term2) in itertools.combinations(ham, 2):
        if not _PauliTerms_commute_trivially(term1, term2):
            commutation_graph.add_edge(term1, term2)

    # color the commutation graph. All terms with one color can be measured
    # simultaneously without the need of ancilla qubits
    color_map = nx.algorithms.coloring.greedy_color(commutation_graph)

    pauli_sum_list = [False] * (max(color_map.values()) + 1)
    for term in color_map.keys():
        if pauli_sum_list[color_map[term]] is False:
            pauli_sum_list[color_map[term]] = PauliSum([term])
        else:
            pauli_sum_list[color_map[term]] += term

    return pauli_sum_list


# ##########################################################################
# Functions to calculate expectation values of PauliSums from bitstrings
# created by a QPU or the QVM
# ##########################################################################


def append_measure_register(program: Program,
                            qubits: List = None,
                            trials: int = 10,
                            ham: PauliSum = None) -> Program:
    """Creates readout register, MEASURE instructions for register and wraps
    in ``trials`` numshots.

    Parameters
    ----------
    param qubits:
        List of Qubits to measure. If None, program.get_qubits() is used
    param trials:
        The number of trials to run.
    param ham:
        Hamiltonian to whose basis we need to switch. All terms in it must
        trivially commute. No base change gates are applied, if ``None``
        is passed.

    Returns
    -------
    Program:
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
    This function assumes, that all terms in ``hamiltonian`` commute trivially
    _and_ that the ``bitstrings`` were measured in their basis.

    Parameters
    ----------
    param hamiltonian:
        The hamiltonian
    param bitstrings: 2D arry or list
        The measurement outcomes. One column per qubit.

    Returns
    -------
    tuple (expectation_value, variance/(n-1))
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
            np.var(energies) / (bitstrings.shape[0] - 1))


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
        e, v = sampling_expectation_z_base(ham, bits)
        energies += e
        var += v
    return (energies, np.sqrt(var))


# ##########################################################################
# Functions to calculate PauliSum expectation values from PauliSums
# without having to create the full hamiltonian matrix
# ##########################################################################

H_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
RX_mat = np.array([[1, -1j], [-1j, 1]]) / np.sqrt(2)


def apply_H(qubit: int, n_qubits: int, wf: np.array) -> np.array:
    """Apply a hadamard gate to wavefunction `wf` on the qubit `qubit`"""
    wf = wf.reshape([2] * n_qubits)
    einstring = "YZ," + ascii_letters[0:n_qubits - 1 - qubit]
    einstring += "Z" + ascii_letters[n_qubits - 1 - qubit:n_qubits - 1]
    einstring += "->" + ascii_letters[0:n_qubits - 1 - qubit] + "Y"
    einstring += ascii_letters[n_qubits - 1 - qubit:n_qubits - 1]
    out = np.einsum(einstring, H_mat, wf)
    return out.reshape(-1)


def apply_RX(qubit: int, n_qubits: int, wf: np.array) -> np.array:
    """Apply a RX(pi/2) gate to wavefunction `wf` on the qubit `qubit`"""
    wf = wf.reshape([2] * n_qubits)
    einstring = "YZ," + ascii_letters[0:n_qubits - 1 - qubit]
    einstring += "Z" + ascii_letters[n_qubits - 1 - qubit:n_qubits - 1]
    einstring += "->" + ascii_letters[0:n_qubits - 1 - qubit] + "Y"
    einstring += ascii_letters[n_qubits - 1 - qubit:n_qubits - 1]
    out = np.einsum(einstring, RX_mat, wf)
    return out.reshape(-1)


def base_change_fun(ham: PauliSum, qubits: List[int]) -> Callable:
    """
    Create a function that applies the correct base change for ``ham``
    on a wavefunction on ``n_qubits`` qubits.
    """
    n_qubits = len(qubits)

    # returns the correct base change function for `qubit`.
    # Make this is a nextra function, because it allows better
    # control flow via `return`
    def _base_change_fun(qubit):
        for term in ham:
            if term[qubit] == 'X':
                return functools.partial(apply_H,
                                         qubits.index(qubit),
                                         n_qubits)
            if term[qubit] == 'Y':
                return functools.partial(apply_RX,
                                         qubits.index(qubit),
                                         n_qubits)
            return None

    funs = []
    for qubit in ham.get_qubits():
        next_fun = _base_change_fun(qubit)
        if next_fun is not None:
            funs.append(next_fun)

    def out(wf):
        return functools.reduce(lambda x, g: g(x), funs, wf)

    return out


def kron_eigs(ham: PauliSum, qubits: List[int]) -> np.array:
    """
    Calculate the eigenvalues of `ham` ordered as a tensorproduct
    on `qubits`. Each qubit should be acted on with the same operator
    by each term or not at all.
    """
    diag = np.zeros((2**len(qubits)))
    for term in ham:
        out = term.coefficient.real
        for qubit in qubits:
            if term[qubit] != 'I':
                out = np.kron([1, -1], out)
            else:
                out = np.kron([1, 1], out)
        diag += out

    return diag


def wavefunction_expectation(hams_eigs: List[np.array],
                             base_changes: List[Callable],
                             hams_squared_eigs: List[np.array],
                             base_changes_squared: List[Callable],
                             wf: np.array) -> Tuple[float, float]:
    """Compute the exp. value and standard dev. of ``hams`` w.r.t ``wf``.

    Parameters
    ----------
    hams_eigs: List[np.array]
        A list of arrays of eigenvalues of the PauliSums in the
        commuting decomposition of the original hamiltonian.
    base_changes:
        A list of functions that apply the neccesary base change
        gates to the wavefunction
    hams_squared_eigs: List[np.array]
        The same as ``hams``, but for the square of ``ham``.
    base_changes_squared:
        The same as ``base_changes``, but for the square of ``ham``.
    wf:
        The wavefunction whose expectation value we want to know.

    Returns
    -------
    Tuple[float, float]:
        A tuple containing the expectation value of ``ham`` and the expectation
        value of ``ham**2``.
    """

    energy = 0
    for eigs, fun in zip(hams_eigs, base_changes):
        wf_new = fun(wf)
        probs = np.abs(wf_new)**2
        energy += probs@eigs

    energy2 = 0
    for eigs, fun in zip(hams_squared_eigs, base_changes_squared):
        wf_new = fun(wf)
        probs = np.abs(wf_new)**2
        energy2 += probs@eigs

    return (energy, energy2)
