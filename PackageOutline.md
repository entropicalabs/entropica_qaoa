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

Ewan's comments:
----------------
If I understand what you mean, I think we should just keep to scipy for now (rather than
trying to include optimisers of our own making that we haven't properly researched yet).
That said, we could consider the option of using other optimisers developed by the ML community
that are not present in scipy.optimize. Joaquin has looked a bit into this recently, we can ask him.

JL's answer:
------------
Agreed, for now we should just use scipy optimizers. I just wanted to structure
the code in such a way, that we can easily switch out the optimizer later for one
that controls sample size and / or takes into account estimates for the standard
deviation. So far I just wrote a wrapper for the scipy optimizer.

EM:
---
Cool!

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

Ewan:
-----
I'm not sure I fully understand the intended use of the abstract cost function here. Would the qvm and qc ones below be a subclass of this abstract one,
so that they are passed to the optimiser through it?

Also, in the methods below, we might want the user to be able to specify an arbitrary intial state (not just the global ground state, or the
equal superposition over all bit strings.) In Grove right now, the QAOA method allows for this, with the initial state prepared by specification of a program
to take you there. In turn, Grove has a CreateArbitraryState method, which returns that program. We could make direct use of this, if the package dependencies are not too messy.

JL:
---
I am also not sure, if an abstract cost_function here is the right thing to do.
What I want, is to give an example of what the required signature for the cost
function will be. Also most of the times we will actually want a "cost function
factory" that takes e.g. a hamiltonian and a QVM connection and then creates a cost
function that can be passed to the optimizer.

About the initial state:
The preparation of an arbitrary initial state would IMHO simply be part of the
Ansatz preparation routine. So in pseudo_code:
    >>> def prepare_ansatz(theta):
    >>>     prepare_state()     # non paramatric preparation of an initial state
    >>>     vqe_circuit(theta)  # parametric circuit.
Now the `prepare_state()` part could simply be the code `CreateArbitraryState` method
you mentioned. But good point!

EM:
--
Let's discuss the optimiser question more on Skype.
About the prepare_state() method, that sounds good.

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

"""
Ewan:
----
Yep, it would be very cool to include the Fourier parametrisation!
Another thing I'd like to build into this (perhaps not for this Forest release, but later) is the adiabatically assisted method that Jose Ignacio's team 
has developed. See here: https://arxiv.org/abs/1806.02287
"""

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

Ewan's comments:
----------------
That's a very good point about qubit placeholders! Let's bear it in mind.
When you ask "what do we use for the graphs", are you referring to visualisation tools?
For that, there is networkx.

JL:
---
Looking at the description of networkx it looks, like that is exactly what we need.
It can use anything as nodes, so especially also QubitPlaceholders.
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

		Ewan's comments:
		----------------
		I assume the input to random_hamiltonian comes from random_graph?
		Let's keep things simple at 2 qubit terms for now, and diagonal in the Z basis.
		By normalised Hamiltonians, do you mean that the spectral range is always the same?

		Other things we could potentially easily include, hinting towards applications and features that researchers might possibly want to explore:

		1) Methods for generating "clustered" data sets with specific properties
		(for instance, two Gaussian-distributed clusters with centre points separated by some specified distance)
		The coupling coefficients in H_cost would then be a function of the Euclidean distance between the points,
		as in Rigetti's unsupervised learning paper of 2017.

		2) Methods for generating Hamiltonians with desired energy landscape features. This might be useful for studying
		how QAOA performs in extreme/pathological/specialised problem instances. For instance, one might want to study the case where
		a specified small number of bitstrings have very low energies, and all others have random but higher energies. The idea here would be to easily
		generate cases where there are a (small) number of deep and narrow minima, i.e. hard problem instances.
		We could also allow this to be specified statistically - eg with there being some well-defined average bitstring energy,
		but with a Gaussian (or other distribution) variance around it.

		3) Methods for generating common types of networks, e.g. Erdos-Renyi, Barbasi, geometric random,...
		[Although maybe keep this for a future release, when we have investigated these graphs better ourselves]

		JL:
		---
		I was actually thinking about more general hamiltonians with more than 2 qubit terms
		and not neccessarily diagonal in the Z-Basis. Of course for QAOA we only need
		diagonal 2 qubit hamiltonians, but I am a fan of having code that isn't more
		specialized than neccesary.

		To get a hamiltonian from a graph I would rather have a function called
		`hamiltonian_from_graph(graph)`, making the distinction clear that there is a
		mapping from graphs to hamiltonians, but not neccesarily the other way around.

		I.e. put all of the functionality you suggest into the random_graph function
		
		EM:
		---
		Sounds good :)
		
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

Ewan's comments:
----------------
I don't know if I really understand the two suggestions following the 'Or' above,
perhaps you can explain them to me on Skype. But what I naively had in mind was more the
object.plot() approach. The object would be an instance of QAOA, with specified cost function,
p value, and gammas and betas. So this visualisation module will be a subclass of QAOA, I think.


JL:
---
Agreed, the object.plot() approach is probably cleaner and easier to extend when adding
e.g. more QAOA parameter classes.
"""

def plot_qaoa_energy_landscape(hamiltonian, parameters):
    """
    Plot the energy landscape of QAOA when holding all parameters fixed except for two.
    Should be nicely integrated with our parameter classes

    :param hamiltonian:  self explanatory
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


"""
Other functionalities to possibly include (Ewan's suggestions)
--------------------------------------------------------------
"""

def PlotBareEnergyLandscape():
    """
    Simply plots the energy of the possible bitstrings, as determined by the Cost Hamiltonian.
    (ie this has nothing to do with the continuous parametrisation via betas and gammas, it is merely the discrete set of energy eigenvalues of H_cost)

		JL's comment
		------------
		So basically just plot the diagonal of the hamiltonian?
		
		EM:
		---
		Yep, plain and simple. But we can build on this, and add more interesting functionality. 
		For example, we could superimpose the probability distribution across the bistrings, so you can see how the peaks in probability correspond to the 
		troughs in energy (hopefully!). You had something similar in your notebook examples when you were exploring the first examples of random Hamiltonians.
		We could even make this an animation (video), so you can see how the distribution changes as the QAOA progresses either (a) during its search for the optimal parameters,
		or (b) when the optimal parameters have already been found, and you simply apply that optimal path through the steps from 1 up to p.
		
		"""

def PlotParametricVariance(parameters2vary, parameter_ranges):
    """
    Plot the landscape of the variance in the energy for the specified parameters.
    This can directly compute from the wavefunction, or also by sampling from the output state.

		JL's comment:
		-------------
		So the same interface as `plot_qaoa_energy_landscape` but plotting the
		variance instead? If yes, one might want think about putting that functionality
		together.
		
		EM:
		---
		Exactly.
		
    """

def PlotOptimalTrajectory(parameters, hamiltonian):
    """
    Plot the path through the landscape of the specified variables that is followed by the optimal trajectory found by QAOA.

		JL's comment:
		-------------
		I am not sure, I understand what you mean. Can you make a quick sketch as an
		example? I can prepare the axes for  you :D

		 ^
		y|
		l|
		a|
		b|
		e|
		l|
		 |
		 .------------------------------->
		 		 	   ylabel
		 		 	   
		 EM:
		 ---
		 
		 Acutally, here what I had in mind was just to superimpose two things:
		 - First, plot the energy landscape for a chosen pair of parameters of interest.
		 - Once you have the optimal parameters found by QAOA, trace the trajectory of the QAOA path in the subspace of the chosen pair of parameters, 
		 by explicitly showing the locations that are visited on the landscape as the intial state is transformed to the final state through the different
		 QAOA steps from step 1 up to step p.
		 		 	   
		 The idea is to plot something like this: https://cdn-images-1.medium.com/max/1600/1*f9a162GhpMbiTVTAua_lLQ.png
		 		 	   
    """

def PlotHessianEigenvalues():
    """
    This may or may not be useful, and may be difficult to implement.
    The eigenvalues of the Hessian should somehow give an idea of how non-convex the landscape is at given parameter values.
    The paper "Visualizing the Loss Landscape of Neural Nets" may serve as a guide.
    The Hessian of the unknown loss function is determined by automatic differentiation, which in itself would need to be implemented (hence needs more work).

		JL's comment:
		-------------
		Interesting idea, but very hard to implement via automatic differentation,
		since we don't have access to the backend of the wavefunction simulator,
		so we can't really implement automatic differentiation. How about finite
		differences?
		
		EM:
		---
		
	    Honestly, I haven't thought too much about how one would do the automatic differentiation here. But, yes, one could just take the naive
	    finite difference approach, and evaluate the cost function at \theta + \epsilon, and \theta - \epsilon, then get the gradient that way. 
	    In any case, I think this one might be a little ambitious for now, and I am not enitrely sure of the value it brings (in particular vs. 
	    the work it would take to implement well). I'd like to find a clear example where it could lead to some specific insight, and then understand
	    how that insight could be applied. This is therefore more of a research thing, so we can build it for ourselves at some point, but perhaps not
	    release it on Forest for now.
		
    """

def PlotEquivalentAnnealingPath():
    """
    For a given QAOA instance and a set of angles (gamma and beta), this will plot the eigenstate populations as a function of time by converting the QAOA pulse sequence into an
    equivalent annealing path, just as JL has done in his work following the method from the Lukin paper.

		JL's comment: 👍
    """
```
