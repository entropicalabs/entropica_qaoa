# Renaming Suggestions
 - QAOAParameter classes:
   [JL] Currently we do e.g.
    `from qaoa.parameters import StandardWithBiasParams` which sounds very redundant if your read it out loud. Also the names are very long... Also some of the names (`General` and `AlternatingOperators` e.g.) are not very descriptive. Any suggestions for better names, for these? Drop the `*QAOAParameterers` in the names altogether? Shorten `*QAOAParameters` to `*Params`?
   [EM] We were thinking `standard`, `placeholder` and `extended` for `Farhi`, `Farhi++` and `General`, respectively. For `placeholder` we still need a proper word that conveys something between the two extremes. 
        We have an idea for a nice graphic to try to convey the meanings, but essentially we have to balance the naming to be informative but not too technical. As Cooper suggested, a good way is to offload part of the challenge to the notebooks and documentation, and establish whatever
        the convention is there.
    [JL] Will do `StandardParams`, `PlaceholderParams`, `ExtendedParams`, `AdiabaticParams` and `FourierParams` then. 

 - `qaoa_params.timesteps`:
   [JL] In all QAOAParamter classes the number of timesteps is currently called `timesteps`. Should we rename it to either simply `p` (QAOA convention) or `n_timesteps` (it is just a number, not a list of timesteps)?
   [EM] I think `steps` would be a reasonable convention. It's descriptive and short, and it's also what Rigetti use in Grove now. (Alternatively, `n_steps`?)
   [JL] I will do `n_steps` then.

 - `adiabatic_timesteps_params._t`
   [JL] Currently we call the total annealing time in `AdiabaticTimestepsQAOAParams` simply `._T`. Is maybe `._annealing_time` a more self-explanatory name?
   [EM] Sure, `annealing_time` seems fine. Perhaps it could be shortened to `anneal_time`.
   [JL] `annealing_time` it is then.

 - `qaoa.cost_functions`:
   [JL] exactly the same goes for the cost_function classes. Additionally PEP8 suggests to name classes that have a `.__call__` method in snake_case, because they behave like functions.

 - `hamiltonian_list_expectation_value`:
   [JL] -> `measurement_expectation` is shorter and (arguably) a bit more concise
   [EM] The name here needs to convey that it's a list of expectation values of different Hamiltonians (if I have understood correctly). How about `expectation_value_list`?
   [JL] Haha, that is exactly the problem. It returns _one_ expectation value calculated over a list (sum) of PauliSums. Has to be done this way, because I can't neccesarily all terms in a PauliSum simultaneously. I will do `sampling_expectation`

 - `hamiltonian_expectation_value`:
   [JL] -> `measurement_expectation_z_base` should have the same name as above, with sth indicating, that this function is limited to diagonal hamiltonians
   [EM] Why not just `expectation_value` for this one?i
   [JL] See previous one. I will call this one `sampling_expectation_only_z` (not the shortest name, but it isn't part of the public interface anyway)

 - `create_random_hamiltonian`, `create_...`:
   [JL] Drop the `create` part in most of those? It is kind of obvious, that these functions will return you e.g. a random hamiltonian. I guess this applies to all function names that start with a somewhat redundant verb like `create` or `get`, where the rest of the name already tells you what will be returned.
   [EM] Generally agreed about dropping create, get. Other less obvious verbs such as 'plot' should stay, though.
   [JL] Perfect 

# All top level names
Here a list of all the top-level(!) names in our packages sorted by length.
If someone knows how to recursively get _all_ names defined by us, feel free to update this list 

['hamiltonian_list_expectation_value',
 'StandardWithBiasParams',
 'AnnealingParams',
 'hamiltonian_expectation_value',
 'address_qubits_hamiltonian',
 'create_gaussian_2Dclusters',
 'create_random_hamiltonian',
 'PrepareAndMeasureOnWFSim',
 'create_circular_clusters',
 'prepare_sweep_parameters',
 'append_measure_register',
 'commuting_decomposition',
 'measurement_base_change',
 'QAOACostFunctionOnWFSim',
 'PrepareAndMeasureOnQVM',
 'AbstractParams',
 'hamiltonian_from_edges',
 'WavefunctionSimulator',
 'ExtendedlParams',
 'QAOACostFunctionOnQVM',
 'FourierParams',
 'QAOAParameterIterator',
 'create_networkx_graph',
 'AbstractCostFunction',
 'make_qaoa_memory_map',
 'prepare_qaoa_ansatz',
 'distances_dataset',
 'QubitPlaceholder',
 'QuantumComputer',
 'scipy_optimizer',
 'MemoryReference',
 'address_qubits']
