# Major ToDo
 - Fix non-diagonal hamiltonians in PrepareAndMeasureOnQVM
   We have two options here: Use the pyquil.paulis.commuting_sets()
   and then more complicated multi qubit controlled gates to measure multi
   qubit terms. Or check equality of the terms in each qubit. Then one-qubit base
   changes do the job.
 - Fix QubitPlaceholders in QAOACostFunction*. Pass a Qubit mapping? Automatically,
   create one, if the hamiltonian contains placeholders? Check if QVMConnection
   can report possible physical Qubits!

# Small things to fix
 - Fix QubitPlaceholders() in VQE cost function
 - Fix QubitPlaceholders() in QAOA cost function
 - Check qaoa.cost_function tests for sanity

