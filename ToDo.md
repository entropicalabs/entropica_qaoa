# Major ToDo
 - Fix non-diagonal hamiltonians in PrepareAndMeasureOnQVM
   We have two options here: Use the pyquil.paulis.commuting_sets()
   and then more complicated multi qubit controlled gates to measure multi
   qubit terms. Or check equality of the terms in each qubit. Then one-qubit base
   changes do the job.
 - Fix QubitPlaceholders in QAOACostFunction * . Pass a Qubit mapping? Automatically,
   create one, if the hamiltonian contains placeholders? Check if QVMConnection
   can report possible physical Qubits!
 - Check all the QAOAParameter docstrings

# Small things to fix
 - Fix QubitPlaceholders() in VQE cost function
 - Fix QubitPlaceholders() in QAOA cost function
 - Check qaoa.cost_function tests for sanity
 - add parameter logging to the cost_functions

# Things to implement
  - utilities.hamiltonian_from_edges
  - utilities.create_random_hamiltonian_fixed_topology

# qaoa.parameters refactoring
 - rename `constant_parameters` to `hyperparameters` (in the arguments to `__init__` and the `__repr__`)
 - rename `variable_parameters` to `parameters` (in the arguments to `__init__` and the `__repr__`)
 - rename `betas`, `gammas_singles` and `gammas_pairs` to `circuit_angles_betas`, ...
 - rename `.from_hamiltonian()` to `.default_params_from_hamiltonian()` or `.linear_ramp_from_hamiltonian()`
 - make the `parameters` accesibly via @property and update the logic in `.update()` and `.update_variable_parameters()`
 - clean up the redundancies in the `QAOAParamters.__init__`, that the user has to pass `hamiltonian` _and_ `qubits_singles`... 
 - move the hyperparameters and parameters into two dicts?
   
