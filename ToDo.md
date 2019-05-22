# Major ToDo
 - [ ] Fix non-diagonal hamiltonians in PrepareAndMeasureOnQVM
   - [ ] Implement `_hamiltonionians_commute_trivially()`
   - [ ] Implement `make_commutation_graph`
   - [ ] Get graph coloring algorithm
   - [ ] ...
 - [x] Fix QubitPlaceholders in QAOACostFunction * . Pass a Qubit mapping? Automatically,
   create one, if the hamiltonian contains placeholders? Check if QVMConnection
   can report possible physical Qubits!
 - [ ] Check all the QAOAParameter docstrings

# Small things to fix
 - [x] Fix QubitPlaceholders() in VQE cost function
 - [x] Fix QubitPlaceholders() in QAOA cost function
 - [ ] Check qaoa.cost_function tests for sanity
 - [ ] add parameter logging to the cost_functions

# Things to implement
  - [ ] utilities.hamiltonian_from_edges
  - [ ] utilities.create_random_hamiltonian_fixed_topology

# qaoa.parameters refactoring
 - [x] rename `constant_parameters` to `hyperparameters` (in the arguments to `__init__` and the `__repr__`)
 - [x] rename `variable_parameters` to `parameters` (in the argumentr to `__init__` and the `__repr__`)
 - [x] rename `betas`, `gammas_singles` and `gammas_pairs` to `x_rotation_angles`, ...
 - [x] rename `.from_hamiltonian()` to `.linear_ramp_from_hamiltonian()`
 - [x]make the `x_rotation_angles` etc accesibly via @property
   - [x] delete `update_variable_parameters()` everywhere
      - [x] delete the commented blocks 
   - [x] rename `update()` to `update_from_raw()`
 - [x] clean up the redundancies in the `QAOAParamters.__init__`, that the user has
   to pass `hamiltonian` _and_ `qubits_singles`
   How? Make him pass `hamiltonian` regardless of the parametrization.
 - [?] rename `self.timesteps` everywhere to `self.p` or `self.ntimesteps`?
 - [?] rename `self._T` to `self.annealing_time` in `AdiabaticTimestepsQAOAParameters`? 
 - [ ] check the documentation and input for validity
 - [x] print coefficients and number of timesteps in `__repr__`
 - [x] convert all the parameter lists to numpy arrays


