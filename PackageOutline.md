A rough suggestions, how we could structure our QAOA package for Rigetti
Forest, such that we can mostly use the code already produced by Ewan and JL
while also keeping it all nice and modular.

# VQE Module
A skeleton of a VQE module. We only provide templates for optimizer and cost
functions here that we later use in the QAOA module. The idea is, that if
someone wants to and or time permits, this can be extended to a full blown VQE
module with swappable optimizers and cost_functions.
A typical class of cost_functions, preparing a state and then measuring its
energy w.r.t some hamiltonian is already implemented, since it is exactly the
form we will need for QAOA.


## optimizer.py
```python
"""
Different implementations of the optimization routines of VQE.
For now we just wrap `scipy.optimize.minimize`, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.

TODO: Is it worth the hassle, to write a abstract_optimizer class that
provides a template and one concrete implementation that just wraps
`scipy.optimize.minimize`?
"""

class optimizer(cost_function, params0, epsilon):
    """
    An optimizer class for VQE. Works with noisy cost_functions that take the
    number of shots as an argument and report the uncertainty of the function
    evaluation.
    """
	# cost_function(params, nshots) -> (cost, sigma_cost)
	# some optimizer of noisy functions which take the number of shots as
    # arguments. Optimizes until changes are smaller than epsilon
```


## cost_functions.py
```python
"""
Abstract description of cost_functions that can be passed to
vqe.optimizer.optimizer instances and two concrete implementations that run on
the QVM resp. QC and measure the energy of a prepared state w.r.t to some
hamiltonian.
"""
class abstract_cost_function(qvm=None, return_float=False, log=None):
	"""Template class for cost_functions that are passed to the optimizer
    :param qvm:             A QVM connectio to run the program on. Creates
                            its own, if None is passed
    :param return_float:    return only a float for scipy.optimize.minimize or
                            (E, sigma_E)
    :param log:             log the function values of all calls here, if
                            provided
    """
    def __init__():
        """Set up all the internals"""
				raise NotImplementedError()

		def __call__(params, nshots) -> (cost, sigma_cost):
				# has to have the same signature as cost_function in `optimizer.py`
				raise NotImplementedError()


class prep_and_measure_ham_qvm(abstract_cost_function):
	"""A cost function that prepares a an ansatz and measures its energy w.r.t
       hamiltonian on the qvm
	"""
	def __init__(prepare_ansatz, hamiltonian: PauliSum):
		# prepare_ansatz(params) -> pyquil.quil.Progam
        self.prepare_ansatz = prepare_ansatz
        self.hamiltonian = hamiltonian


	def __call__(params, nshots):
		wf = qvm.Wavefunction(self.prepare_ansatz(params))
        # might want to add noise to E for more realistic cases. Maybe provide
        # noisy and non-noisy version of this
		E = <wf|self.hamiltonian|wf>
		sigma_E = nshots**(-1/2) * (<wf|self.hamiltonian**2|wf> - E**2)
		return(E, sigma_E)


class prep_and_measure_ham_qc(abstract_cost_function):
	"""A cost function that prepares a an ansatz and measures its energy w.r.t
       hamiltonian on the qc
	"""
	def __init__(self, prepare_ansatz, hamiltonian: PauliSum):
        self.hamiltonian = hamiltonian
        self.prepare_ansatz = prepare_ansatz
		self.hamiltonian_commuting_terms = decompose_commuting(hamiltonian)
		self.exes = []
		for commuting_term in self.hamiltonian_commuting_terms:
			prog = append_base_change_and_measure(self.prepare_ansatz,
                                                   commuting_term)
			exes.append(qc.compile(progs))

	def __call__(self, params, nshots):
        # make_memory_map creates a memory_map from the parameters that can be
        # passed to qc.run.
		memory_map = make_memory_map(params)
		bits = qc.run(self.exes, )
        # calculate_expectation_value takes a hamiltonian and bitstrings and
        # calculates the expectatation value and its standard_deviation
		(E, sigma_E) = calculate_expectation_value(hamiltonian, bits)
		return (E, sigma_E)
```


# QAOA module
The actual QAOA module.
It provides cost_functions for the VQE module that can be passed to the
vqe.optimizer instances.
It also provides our different classes of QAOA parameters that play nicely
with qaoa_cost_function_qvm and qaoa_cost_function_qc

## cost_functions.py
```python
"""
Concrete implementations of cost_functions that can be passed to
vqe.optimizer.optimizer instances and are the cost functions for QAOA.

TODO: Take qaoa out of all the function names, since it all sits in a module
called qaoa?
"""
class qaoa_cost_function_qvm(prep_and_measure_ham_qvm):
    """
    A cost function that inherits from prep_and_measure_ham_qvm and implements
    the specific prepare_ansatz from QAOA
    """
    def __init__(hamiltonian, p, tau):
        self.prepare_ansatz = prepare_qaoa_ansatz_qvm(hamiltoninan, p, tau)
        #...
    # __call__() is inherited


class qaoa_cost_function_qc(prep_and_measure_ham_qc):
    """
    A cost function that inherits from prep_and_measure_ham_qc and implements
    the specific prepare_ansatz from QAOA
    """
    def __init__(hamiltonian, p, tau):
        self.prepare_ansatz = prepare_qaoa_ansatz_qc(hamiltonian, p, tau)
         # because all the terms commute already
        self.hamiltonian_commuting_terms = [hamiltonian]
        self.exes = append_base_change_and_measure(self.prepare_ansatz,
                                                    hamiltonian)
    # __call__() is inherited


def prepare_qaoa_ansatz_qvm(hamiltonian, p, tau):
    """
    :return: A function that takes qaoa_params and creates a pyquil.Program
    """
    # implementation the same as in the code already there


def prepare_qaoa_ansatz_qc(hamiltonian, p, tau):
    """
    :return: A function that takes qaoa_params and creates a parametric
             pyquil.Program
    """
    # implentation the same as in the code already there
```

## parameters.py
```python
"""
Essentially the same file, that we already wrote
"""
class abstract_qaoa_params():
    """
    The same as already written
    """
    # Force to include a calculate_initial_parameters function?


class general_qaoa_parameters(abstract_qaoa_parameters):
    """The same as already written"""


# include this into the general_qaoa_parameters class?
def calculate_initial_parameters_general(hamiltonian, reg, p, tau=None):
    """
    Once again the same as is already there.
    Possibly rethink the inclusion of reg? Make it optional and if None is
    passed use hamiltonian.get_qubits()?
    """


# rename to classic_qaoa_parameters?
class alternating_operators_qaoa_parameters(abstract_qaoa_parameters):
    # obvious. Same as is there

def calculate_initial_parameters_alternating_operators(....):
    """obvious"""

# also fourier and adiabatic_timesteps just as above
```


## utilities.py
```python
"""
Some more useful utilities to experiment with QAOA like convenience functions
to create the hamiltonian corresponding to a given graph or to create graphs
of fixed size and/or degree.

TODO:
What do we use for the graphs? Is there already a good implementation in
python we can use? Can it use arbitrary hashable objects as vertices?
(convenient, if one wants to use pyquil.quilatom.Qubit or
pqyuil.quilatom.QubitPlaceholder as vertices)
"""

def qaoa_hamiltonian_from_graph(graph) -> PauliSum:
    """Takes a graph and returns the corresponding hamiltonian for QAOA

    :rtype: pyquil.paulis.PauliSum       # because we want to integrate nicely
    """
    # yup, needs to be implemented


# Just a proposal, how such a random graph generating function could look like
def random_graph(n_vertices: int,
                 n_edges = 0: int,
                 degree = 0: int,
                 uniform_weights = False):
    """
    Create a random graph on n vertices with fixed number of edges and/or degree

    :param n_vertices:  Number of vertices
    :n_edges:           Number of edges. Not necessary if degree is passed
    :degree:            Degree of the graph. Defaults to 2, if none is passed
                        (because that is the maximum degree natively possible
                        on the Aspen QPU)
    :uniform_weights:   Random or uniform weights on the edges?

    :return:            A random graph with the specified properties
    """


def random_hamiltonian(qubits, nterms = None, Type = "free") -> pyquil.paulis.PauliSum:
    """
    Create a random hamiltonian on nqubits qubits.
    TODO: Decide, what Types make sense to support. Hamiltonians only diagonal
    in the computational basis? Hamiltonians with at most 2 qubit terms?
    Normalized hamiltonians?

    :param nqubits: (int or list of qubits) Qubits to have the
    hamiltonian on
    :param nterms:  (int) Number of terms in the hamiltonian
    :param Type:    Type of hamiltonian to create. See TODO
    :rtype:         (pyquil.paulis.PauliSum) The hamiltonian
    """
```


## visualization.py
```python
"""
Some convenience functions to easily visualize the workings of QAOA

TODO:
Do we want to write a plot_object() function for each object that can be
sensibly plotted or do we want to write object.plot() for each object, that
can be sensibly plotted?
The former is easier to write and discover at first while the latter clutters
the namespace less (no need to remember ten different plot_something()
functions) and is probably easier to maintain. If changes are made to an
object the corresponding plot function is not in a different file.

Or: Split the difference: Let the objects plot themselves, but put
visualizations that aren't tied to a particular object in here?

Or: Put plot_qaoa_parameters function here, that simply calls parameters.plot()
"""

def plot_qaoa_energy_landscape(hamiltonian, parameters):
    """
    Plot the energy landscape of QAOA when holding all parameters fixed except for one.
    Should be nicely integrated with our parameter classes

    :param hamilttonian: self explanatory
    :param parameters:   a subtype of qaoa.abstract_qaoa_parameters
    """
    # see, what of Ewans Code can be used here


# let them possibly plot themselves?
def plot_qaoa_parameters(parameters):
    """
    Plot the optimal QAOA parameters found in optimization

    :param parameters:   a subtype of qaoa.abstract_qaoa_parameters
    :rtype:              matplotlib axes object? None?
    """
    # implementation suggestion:
    parameters.plot()


def vqe_optimization_stacktrace_plot(ax=None):
    """
    Plot the cost_function values of a VQE run (and by extension of a QAOA run)
    :TODO:  Decide, whether to create the log in the cost_function or in the
            optimizer. Does scipy.optimize.minimize allow logging of function
            calls?
    """
    # implementation details to follow


# let the graphs plot themselves?
def plot_graph(graph):
    """
    Creates a nice plot of the graphs used in QAOA
    """
    # implementation details to follow
    
-----------------------------------------------
Other functionalities to possibly include (Ewan's suggestions)

def PlotBareEnergyLandscape():

    """
    Simply plots the energy of the possible bitstrings, as determined by the Cost Hamiltonian. 
    (ie this has nothing to do with the continuous parametrisation via betas and gammas, it is merely the discrete set of energy eigenvalues of H_cost)
    """
    
def PlotParametricVariance(parameters2vary,parameter_ranges):

    """
    Plot the landscape of the variance in the energy for the specified parameters.
    This can directly compute from the wavefunction, or also by sampling from the output state. 
    
    """
    
def PlotOptimalTrajectory(parameters,hamiltonian):

    """
    Plot the path through the landscape of the specified variables that is followed by the optimal trajectory found by QAOA.
    """
    
def PlotHessianEigenvalues():

    """
    This may or may not be useful, and may be difficult to implement. 
    The eigenvalues of the Hessian should somehow give an idea of how non-convex the landscape is at given parameter values.
    The paper "Visualizing the Loss Landscape of Neural Nets" may serve as a guide.
    The Hessian of the unknown loss function is determined by automatic differentiation.
    """
    
```
