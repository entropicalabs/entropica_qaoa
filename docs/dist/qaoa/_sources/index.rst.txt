.. Entropica QAOA documentation master file, created by
   sphinx-quickstart on Tue Apr 23 00:13:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EntropicaQAOA
========================

EntropicaQAOA is a modular package for the quantum approximate optimisation algorithm (QAOA) built on top of Rigetti's
`Forest SDK <https://www.rigetti.com/forest>`_. 

Features
--------

 - Multiple built-in parametrisations for QAOA.
 - `NetworkX <https://networkx.github.io/>`_ integration for the treatment of
   graph theoretic problems, and `Pandas <https://pandas.pydata.org/>`_ integration for data importing.
 - Convenient optimiser logging tools.
 - Ability to run more general parametric circuits using the VQE library.
 - Full native support for `Rigetti's QVM and QPUs <https://www.rigetti.com/qpu>`_. Run your experiments on real Quantum Computers!
 - Modular code base that allows the user to customise and modify different components.
 - Extensive examples explaining the usage of EntropicaQAOA in detail.


Installation
------------

If you don't have them already, you should first install Rigetti's pyQuil package, as well as their QVM and Quil Compiler. For instructions on how to do so, see Rigetti's documentation `here <http://docs.rigetti.com/en/stable/start.html>`_.

You can install EntropicaQAOA using `pip <https://pip.pypa.io/en/stable/quickstart/>`_:

.. code-block:: bash

   pip install entropica_qaoa

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade entropica_qaoa

If you want to run the Demo Notebooks, you will additionally need to install `scikit-learn` and `scikit-optimize`:

.. code-block:: bash

   pip install scikit-learn && pip install scikit-optimize

You can also install EntropicaQAOA directly from GitHub. In your desired local directory, first clone the repository

.. code-block:: bash

   git clone https://github.com/entropicalabs/entropica_qaoa.git

then move into the ``entropica_qaoa`` directory, and run pip:

.. code-block:: bash

   cd entropica_qaoa
   pip install -e


Testing the installation
------------------------

Run the following short code to verify that EntropicaQAOA has been successfully installed. Here, we create a simple Hamiltonian 
and evaluate its expectation value with respect to the wavefunction produced by a QAOA circuit. We choose the ``StandardParams`` parametrisation,
and initialise the parameter values by analogy to a quantum annealing process with a linear schedule function. These features are explained in depth in other sections of 
the documentation. 

.. code-block:: python

   # pyquil imports
   from pyquil import Program
   from pyquil.api import WavefunctionSimulator
   from pyquil.paulis import PauliSum, PauliTerm

   # EntropicaQAOA imports
   from entropica_qaoa.qaoa.parameters import StandardParams
   from entropica_qaoa.qaoa.cost_function import QAOACostFunctionOnWFSim

   # create a hamiltonian on 3 qubits with 2 coupling terms and 1 bias term
   Term1 = PauliTerm("Z", 0, 0.7)*PauliTerm("Z", 1)
   Term2 = PauliTerm("Z", 0, 1.2)*PauliTerm("Z", 2)
   Term3 = PauliTerm("Z", 0, -0.5)
   ham = PauliSum([Term1,Term2,Term3])

   p = 3 # QAOA p parameter (number of timesteps)
   params = StandardParams.linear_ramp_from_hamiltonian(ham,p) # initialise a set of StandardParams

   # Build the cost function and compute its value with the circuit parameters initialised above
   cost_fun = QAOACostFunctionOnWFSim(ham,params)
   res = cost_fun(params.raw())
   print(res)
  
You should find the value of ``res`` to be -1.4664694.

If you installed directly from GitHub, you can also run the full set of software tests. To do so, you will need to install `pytest <https://docs.pytest.org/en/latest/>`_. To speed up the testing, we have tagged tests that require more computational time (~ 5 mins or so)  with `runslow`, and the tests of the notebooks with `notebooks`. The commands are as follows:

 - ``pytest`` runs the default tests, and skips both the longer tests that need heavier simulations, as well as tests of the Notebooks in the ``examples`` directory.
 - ``pytest --runslow`` runs the the tests that require longer time.                              
 - ``pytest --notebooks`` runs the Notebook tests. To achieve this, the notebooks are converted to python scripts, and then executed. Should any errors occur, this means that the line numbers given in the error  messages refer to the lines in `<TheNotebook>.py`, and not in `<TheNotebook>.ipynb`.
 - ``pytest --all`` runs all of the above tests.   

The documentation can also downloaded as Jupyter notebooks from our `GitHub page <https://github.com/entropicalabs/entropica_qaoa/tree/master/examples>`_ .


Contributing and feedback
-------------------------

If you have feature requests, or have already implemented them, feel free to open an issue or send us a pull request. 

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you'd like to tell us about, drop us an email at devteam@entropicalabs.com. 


License
-------

EntropicaQAOA is released under the Apache License, Version 2.0.

Contents
========

.. toctree::
   :maxdepth: 3

   notebooks/1_AnExampleWorkflow
   notebooks/2_ParameterClasses
   notebooks/3_AdvancedParameterClasses
   notebooks/4_CostFunctionsAndVQE
   notebooks/5_QAOAUtilities
   notebooks/6_ClusteringWithQAOA

   faq
   changelog


.. toctree::
   :maxdepth: 3
   :caption: API Reference

   vqe_cost_function
   qaoa_cost_function
   parameters
   utilities
   measurelib


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
