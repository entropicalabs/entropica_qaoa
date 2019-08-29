
                **Contents of the example notebooks**

# `QAOA_Quickstart.ipynb`
A Notebook giving a quick overview over the basic functionality of the package that at the same time serves as a summary for the coming notebooks. Should contain:

 - Creation of a hamiltonian from a graph. A note that more graph and
   hamiltonian conversions can be found in `Utilities.ipynb`
 - Creating new parameters of Type `Standard` from this hamiltonian. Then a
   note that more parametrizations and more things you can do with the
   parameters can be found in `QAOA_Parametrizations.ipynb`
 - Creating a QAOA cost function with these parameters and a hamiltonian. Then
   a note that the possibility to run on the QPU instead of the WFsim and more
   sophisticated options for the cost functions are explained in 
   `VQE_and_cost_functions.ipynb`
 - A run with the optimizer and again the note, that the demo of other
   optimizers and more details in `VQE_and_cost_functions.ipynb`


# `QAOA_Parametrizations.ipynb`
This Notebook is twofold (and might be split into two Notebooks): On one hand it explains all parametrizations (`Standard`, `Fourier` and `Annealing`) in more details and on the other hand it shows off different 

## Different parametrizations
 - Explain Standard exactly (with the formulas)
 - Explain Fourier exactly (with the _correct_ formulas)
 - Explain Annealing exactly (with the formulas)

## Working with parametrizations
 - modifying them
 - parameters under the hood
 - parameter iterators

## Parameter creation routines and conversions
 - `.linear_ramp_from_hamiltonian()`
 - `.empty()`
 - `.from_other_parameters()`
 - `.from_AbstractParameters()`

# `VQE_and_cost_functions.ipynb`
This notebook explains the optional arguments of the cost functions based on
VQE. There is a big note, that all of this applies also to QAOA, since that is
only a special instance of VQE.

## Using different optimizers
 - well, demo how to use optimizers other than `scipy.optimize.minimize`

## the other kwargs
 - the `nshots` argument
 - the `scalar_cost_function` argument
 - the `enable_logging` argument

# `Utilities.ipynb`
 - graph creation routines
 - hamiltonian to graph conversions
 - landscape demos?
