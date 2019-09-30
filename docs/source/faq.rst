.. _faq:

Implementation details, conventions, and FAQ
============================================

Sign of the mixer Hamiltonian
    In the original paper on QAOA (`Ref 1 <#references>`__), Farhi `et al` use :math:`\sum_i \hat{X}_i` as
    the mixer Hamiltonian, with the initial state being its maximum eigenstate :math:`\left|+ \cdots +\right>`. 
    In EntropicaQAOA, we instead choose our mixer Hamiltonian to be :math:`-\sum_i \hat{X}_i`, so that the initial state 
    :math:`\left|+ \cdots +\right>` is now its minimum energy eigenstate. Conceptually this makes the analogy to adiabatic
    computing clear, since we seek to transform from the ground state of the mixer Hamiltonian to the ground state of the cost Hamiltonian. 

Implementation of circuit rotation angles
    In quantum mechanics, the basic time evolution operator is :math:`\exp(-iHt)` for a Hamiltonian `H` and total
    evolution time `t`. Generically, in the QAOA mixer Hamiltonian, the operator :math:`-X` is to be applied for a total time 
    :math:`\beta`, which is one of the parameters we seek to optimise. We therefore need to implement the time evolution 
    :math:`\exp(i\beta X)`, which can be achieved using the RX(:math:`\theta`) operator if we set :math:`\theta = -2\beta`. 

    Similarly, the cost Hamiltonian operator :math:`\exp(-i\gamma hZ)` can be implemented via an RZ(:math:`\theta`) rotation, setting
    :math:`\theta = 2\gamma h`. In the functions ``qaoa.cost_function._qaoa_cost_ham_rotation`` and ``qaoa.cost_function._qaoa_mixing_ham_rotation``, you can verify these details.

Where does the factor ``0.7 * n_steps`` in the ``linear_ramp_from_hamiltonian()`` method come from?
    The ``.linear_ramp_from_hamiltonian()`` parameters are inspired by analogy between
    QAOA and a discretised adiabatic annealing process. If we pick a linear ramp annealing schedule, i.e. :math:`s(t) = \frac{t}{\tau}`, where :math:`\tau` is the total
    annealing time, we need to specify two numbers: the total annealing time :math:`\tau` and the step width
    :math:`\Delta t`. Equivalently, we can also specify the total annealing time :math:`\tau` together with
    the number of steps :math:`n_{\textrm{steps}}`, which is also called `p` in the
    context of QAOA. A good discretised annealing schedule has to strike a
    balance between a long annealing time :math:`\tau` and a small step width
    :math:`\Delta t = \frac{\tau}{n_{\textrm{steps}}}`. We have found in numerical
    experiments that :math:`\Delta t = 0.7 = \frac{\tau}{n_{\textrm{steps}}}` strikes a reasonably good balance
    for many problem classes and instances, at least for the small system sizes one can feasibly simulate.
    For larger systems or smaller energy gaps, it might be neccesary to choose smaller values of :math:`\Delta t`

Computation of cost function expecation values
    To compute the expectation value of the cost Hamiltonian on the wavefunction simulator, we have attempted to address a trade-off in two different possible methods. One way is to use Forest's native
    ``sim.expectation(prog,ham)`` method (see `here <http://docs.rigetti.com/en/stable/apidocs/autogen/pyquil.api.WavefunctionSimulator.expectation.html>`__), with ``prog`` being the QAOA circuit and ``ham`` being the cost Hamiltonian (a PauliSum object) of interest. However, this computes the expectation value of each term in the PauliSum individually, and then sums up the results; the runtime can therefore be significant when there are many terms to evaluate. On the other hand, one could instead build the matrix representing the entire cost Hamiltonian, and apply it to the output wavefunction. However, for many qubits this can be very memory intensive, since the Hamiltonian is a :math:`2^n \times 2^n`-dimensional matrix.

    In many problems of interest for QAOA, the cost function is diagonal in the computational basis, and it is therefore sufficient to build only a :math:`2^n`-dimensional vector. If the cost Hamiltonian were to also contain non-commuting terms (e.g. terms proportional to :math:`X`), we could perform a suitable basis change and again measure the expectation with respect to a diagonal matrix (a :math:`2^n` vector) built from the operators in that basis. 

    In EntropicaQAOA, we decompose the cost Hamiltonian (a PauliSum) into sets of operators that commute trivially. Two Pauli products commute trivially if on each qubit both act with the same Pauli Operator, or if either one acts only with the identity. Let's suppose our Hamiltonian contains terms proportional to :math:`Z` and terms proportional to :math:`X`. When working with the wavefunction simulator, we then have two sets of operators, each of which can be represented as a :math:`2^n` vector. Measurement of the terms proportional to :math:`Z` is trivial, and for the terms proportional to :math:`X`, we perform a basis change on the wavefunction and then measure. In order to avoid building a large matrix to execute the basis change, we use the ``einsum`` method in Numpy.

    For computations on the QPU, we again separate the terms into trivially commuting sets, and now the basis change is performed via a suitable rotation on the qubits - e.g. a Hadamard gate, if we wish to measure in the :math:`X` basis.
    

Discrete sine and cosine transforms for the ``FourierParams`` class
    In converting between the :math:`\beta` and :math:`\gamma` parameters of the ``StandardParams`` class, and the `u` and `v` parameters of the 
    ``FourierParams`` class, we use the type II versions of the discrete sine and cosine transformations. These are included in Scipy's fast Fourier 
    transforms module `fftpack <https://docs.scipy.org/doc/scipy-0.14.0/reference/fftpack.html>`_. With the conventions used therein, in EntropicaQAOA the transformations are then given by:

    .. math::

	\gamma_i = 2 \sum_{k=0}^{q-1} u_k
		      \sin\left[
		             (k + 1/2)
    			     (i+1)			
                             \frac{\pi}{p}
		          \right]

	\beta_i = 2 \sum_{k=0}^{q-1} v_k
		      \cos\left[
		            (2k + 1) 
		            i\frac{\pi}{2p}
		          \right]
 
    While these differ from the versions used in `Ref 2 <#references>`__, this is merely a convention.

What is the difference between ``base_numshots`` and ``n_shots`` in ``PrepareAndMeasureOnQVM`` and ``QAOACostFunctionOnQVM``?
    The cost functions created by ``PrepareAndMeasureOnQVM`` and ``QAOACostFunctionOnQVM`` both make use of Quil's
    `parametric program <http://docs.rigetti.com/en/latest/basics.html?programs#parametric-compilation>`_ functionality. This means that the circuit is
    compiled once, before the optimisation starts, and then only the variable parameters are changed by the optimiser. Currently, the number of
    circuit repetitions can only be set once before compilation, via the command `Program.wrap_in_numshots_loop(base_numshots) <http://docs.rigetti.com/en/latest/apidocs/autogen/pyquil.quil.Program.wrap_in_numshots_loop.html>`_. 
    If running on the QVM, this means that the Wavefunction is calculated once, and ``base_numshots`` samples are taken from it. Of course, on the QPU itself the same program has to be run `base_numshots` times.

    Now in collecting statistics, we may want to understand how the number of samples we take affects quantities like expectation values and standard deviations. In Quil's parametric program framework, if we want to look at how statistics change if (say) we double the number of samples, we would need to recompile the program, since the number of samples to be taken is hard-coded. By introducing ``base_numshots``, we can compile the circuit once with a given number of samples to be taken, and simply run the program twice (setting ``n_shots = 2``) to obtain double the number of samples, without the need to re-compile. A further conceivable use case for ``base_numshots`` is in dynamically modifying the number of samples taken during an optimisation process, depending on (say) the observed sample standard deviation for some specific set of parameters.

    Setting ``n_shots = 1`` (the default value) effectively disables this functionality.

References
----------

1. E. Farhi et al, `A Quantum Approximate Optimization Algorithm <https://arxiv.org/abs/1411.4028>`__
2. L. Zhou et al, `Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices <https://arxiv.org/abs/1812.01041>`__ 
