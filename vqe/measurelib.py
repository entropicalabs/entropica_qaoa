"""
Various convenience functions for measurements on a quantum computer or
wavefunction simulator
"""
from pyquil.quil import MEASURE

import numpy as np


def append_measure_register(program, qubits=None, trials=10):
    """Creates readout register, MEASURE instructions for register and wraps
    in trials trials.

    Parameters
    ----------
    param qubits : list
        List of Qubits to measure. If None, program.get_qubits() is used
    param trials : int
        The number of trials to run.

    Returns
    -------
    Program :
        program with the measure instructions appended
    """
    if qubits is None:
        qubits = program.get_qubits()

    ro = program.declare('ro', memory_type='BIT', memory_size=len(qubits))
    for i, qubit in enumerate(qubits):
        program += MEASURE(qubit, ro[i])
    program.wrap_in_numshots_loop(trials)
    return program


def hamiltonian_expectation_value(hamiltonian, bitstrings):
    """Calculates the energy expectation value of ``bitstrings`` w.r.t ``ham``.

    Parameters
    ----------
    param hamiltonian : PauliSum
        The hamiltonian
    param bitstrings : 2D arry or list
        the measurement outcomes. Columns are outcomes for one qubit.

    Returns
    -------
    tuple (expectation_value, standard_deviation)

    Warning
    -------
    Only handles hamiltonians that are sums of products of  Zs!
    """
    # TODO fix this to handle arbitrary hamiltonians
    if bitstrings.ndim == 2:
        energies = np.zeros(bitstrings.shape[0])
    else:
        energies = np.array([0])
    for term in hamiltonian:
        sign = np.zeros_like(energies)
        for factor in term:
            sign += bitstrings[:, factor[0]]
        energies += term.coefficient.real * (-1)**sign

    return (np.mean(energies),
            np.sqrt(np.var(energies)) / np.sqrt(bitstrings.shape[0]))
