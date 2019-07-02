# Major breaking(!) refactoring for qaoa.parameters
We changed some parts of the interface. To make the code run with the new version,
you need to follow the following things:
 - `params.update()` was renamed to `params.update_from_raw()`
 - `params.betas, params.gammas_singles, params.gammas_pairs`
    are renamed to `params.x_rotation_angles, params.z_rotation_angles,
    params.zz_rotation_angles`
 - `params.update_variable_parameters()` is gone. Just delete all occurences
    of it, the class now takes care of the functionality all by itselt. It is
    an adult now.
 - `QAOAParameters.from_hamiltonian()` got renamed to `QAOA_Parameters.linear_ramp_from_hamiltonian`
 - All child classes of `AbstractParams` `__init__`-function have changed signature.
   `hyperparams` is now of the form `(hamiltonian, timesteps , ...)` wherer ... is the total
   annealing time or the q-number of fourier params.
 - All of the `QAOAParameters.linear_ramp_from_hamiltonian` have changed signature. The
   `reg` parameter is gone.
