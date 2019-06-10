# Major ToDo
 - [x] Fix non-diagonal hamiltonians in PrepareAndMeasureOnQVM
   + [x] Implement `_hamiltonionians_commute_trivially()`
   + [x] Implement `make_commutation_graph`
   + [x] Get graph coloring algorithm
   + [ ] Make `measurement_base_change()` part of `append_measure_register()` and add `base=..` or `ham=...` option to `append_measure_register()`
- [x] Fix QubitPlaceholders in QAOACostFunction * . Pass a Qubit mapping?
  + [ ] Automatically, create one, if the hamiltonian contains placeholders?
  + [ ] Check if QVMConnection can report possible physical Qubits!
- [ ] Change `AlternatingParameters` to `FarhiQAOA` or `ClassicalQAOA`
- [ ] Check, that everything runs on the QPU
- [ ] Profile the whole QAOA part and optimize
  + [x] Ask Asad or Cooper for production code for realistic profiling
  + [ ] Get to know `cProfile`
- [ ] Rename long functions / classes
  + [ ] Start Discussion Thread on Slack
- [ ] Add `state_prep_program` to `qaoa.cost_function`
  + [ ] Add `prep_classical_state(bitstring)` to `utilities.py`
- [ ] Check all the QAOAParameter docstrings
  + [ ] Make `__repr__` uniform over all QAOAParameterClasses

# Small things to fix
 - [x] Fix QubitPlaceholders() in VQE cost function
 - [x] Fix QubitPlaceholders() in QAOA cost function
 - [ ] Check qaoa.cost_function tests for sanity
 - [?] add parameter logging to the cost_functions

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
 - [x] clean up the redundancies in the `QAOAParamters.__init__`, that the
       user has to pass `hamiltonian` _and_ `qubits_singles`
       How? Make him pass `hamiltonian` regardless of the parametrization.
 - [ ] check the documentation and input for validity
 - [ ] make `self.betas`, `self.gammas_singles` etc @property instead of attributes
       for automatic input checking (related to above). Check if numpy supports automatic
       broadcasting from i.e. `(6, ) -> (3,2)`
 - [x] print coefficients and number of timesteps in `__repr__`
 - [x] convert all the parameter lists to numpy arrays
 - [?] rename `self.timesteps` everywhere to `self.p` or `self.ntimesteps`?
 - [?] rename `self._T` to `self.annealing_time` in `AdiabaticTimestepsQAOAParameters`? 
 


