.. Entropica QAOA documentation master file, created by
   sphinx-quickstart on Tue Apr 23 00:13:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Entropica QAOA's documentation!
==========================================

Entropica QAOA is a modular QAOA package built on top of Rigetti's
`Forest SDK <https://www.rigetti.com/forest>`_. It enables researchers and
interested students to quickly run their own QAOA instances and easily replace
any component with your own code. You don't like the standard parametrisation
of the QAOA circuit? Use any of our other parametrisations or write your own.
You want to see what happens, if you start in different initial states? Run
your own state preparation program. You wan't to see, how different optimisers
are with noise estimates of expectation values? We can efficiently simulate sampling noise.

Features
--------
 - Full native support for Rigettis QVM and QPU's. Run your experiments on
   real Quantum Computers!
 - Extensive iPython Notebooks that explain the usage of Entropica QAOA in
   detail.
 - Multiple parametrisations for QAOA.
 - NetworkX integration for the treatment of graph theoretic problems.
 - Modular code base that allows the user to customise parts he wants to change
   while being provided with sensible defaults for the other parts.

Installation
------------

If you don't have them already, install first Rigetti's pyQquil package and their QVM and Quil Compiler. For instructions on how to do so, see the Rigetti documentation here: http://docs.rigetti.com/en/stable/start.html

Installation of the ``entropica_qaoa`` package can be performed in a few simple steps.

1. Open terminal and enter the site-packages folder of your preferred Python environment.

   For those with Anaconda installed, the command looks like:

   .. code-block:: bash

      cd /anaconda3/envs/<my-env>/lib/pythonX.Y/site-packages/

   For those unsure of the location of their site-packages folder, you can simply run ``bash pip show <package name>`` and your terminal will display the directory location of your python packages.


2. Clone the repository into your site-packages folder, into a directory called entropica_qaoa:

   .. code-block:: bash

      git clone https://github.com/entropicalabs/entropica_qaoa entropica_qaoa

3. Install the package using pip:

   .. code-block:: bash

      pip install entropica_qaoa


You can now import this package as you would any conda- or pip-installed library!


First Steps
-----------

In *First steps: An example workflow* we show you the basic usage of
EntropicaQAOA.
More advanced examples on working with different parametrizations are found in
*Working with the Parameter classes* and the more advanced features of the cost
functions are explained in *Cost function features and VQE*. To learn, how to
create problem instances faster and convert between different formats you can
read *QAOA utilities*.

If you are not content just reading above linked notebooks you can also download them from our github page at `<https://github.com/entropicalabs/entropica_qaoa/tree/master/examples>`_ .


Contents
========

.. toctree::
   :maxdepth: 3

   faq
   changelog

.. toctree::
   :maxdepth: 3
   :caption: The Demo Notebooks

   notebooks/1_AnExampleWorkflow
   notebooks/2_ParameterClasses
   notebooks/3_AdvancedParameterClasses
   notebooks/4_CostFunctionsAndVQE
   notebooks/5_QAOAUtilities
   notebooks/6_ClusteringWithQAOA


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
