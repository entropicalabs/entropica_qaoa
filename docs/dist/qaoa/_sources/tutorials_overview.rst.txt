Overview of tutorials
=====================

The tutorials in the following sections illustrate the key features of EntropicaQAOA. Here we provide a brief description of each of them, to assist in navigating the examples.

:ref:`1-AnExampleWorkflow` provides a quickstart introduction to some of the main features of the package, in the context of a typical QAOA workflow. Expert users may wish to begin here and refer to the other sections as needed.

:ref:`2-ParameterClasses` provides a comprehensive introduction to the different built-in QAOA parametrisations, one of the main features of the package. It focuses in particular on the ``Standard`` and ``Extended`` parameter classes. The former is the conventional parametrisation, with one angle for each of the mixer and cost Hamiltonians at each timestep. The latter allows all terms to have its own tunable parameter at every time step. This section also describes in detail different options for initialising parameters, and converting between classes. 

:ref:`3-AdvancedParameterClasses` provides details of two additional parametrisation classes - ``Annealing`` and ``Fourier`` - which may be of interest for different purposes.

:ref:`4-CostFunctionsAndVQE` shows how we can use the package for more general variational quantum eigensolver (VQE) problems. It also demonstrates a number of workflow tools, including optimiser logging functions, and utilties that assist in accounting for measurement statistics when working on the wavefunction simulator. In addition, this section also provides an example of how to run a computation on the real QPU (as well as the QVM).

:ref:`5-QAOAUtilities` provides a walk-through of some of EntropicaQAOA's convenience methods for setting up QAOA problems directly from objects of popular libraries such as ``NetworkX`` or ``Pandas``.

:ref:`6-SolvingQUBOwithQAOA` illustrates how QAOA can be used to solve a simple quadratic unconstrained binary optimisation task (QUBO). This class of problem has widespread real-world applications, particularly in fields such as logistics.

Finally, :ref:`7-ClusteringWithQAOA` provides a didactic look at the application of QAOA to perform clustering (via MaxCut) on the Pokemon data set. In particular, this tutorial is geared more towards users from a computer science or data science background.

 


