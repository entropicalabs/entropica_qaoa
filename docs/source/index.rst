.. Entropica QAOA documentation master file, created by
   sphinx-quickstart on Tue Apr 23 00:13:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EntropicaQAOA's documentation!
==========================================

EntropicaQAOA is a modular QAOA package built on top of Rigetti's
`Forest SDK <https://www.rigetti.com/forest>`_. It enables users to quickly run their own QAOA instances and easily replace
any component with their own custom code.

Features
--------
 - Full native support for `Rigetti's QVM and QPUs <https://www.rigetti.com/qpu>`_. Run your experiments on real Quantum Computers!
 - Extensive examples explaining the usage of EntropicaQAOA in
   detail.
 - Multiple parametrisations for QAOA.
 - `NetworkX <https://networkx.github.io/>`_ integration for the treatment of
   graph theoretic problems, and `Pandas <https://pandas.pydata.org/>`_ integration for data importing.
 - Modular code base that allows the user to customise and modify different components.


Installation
------------

If you don't have them already, install first Rigetti's pyQquil package and their QVM and Quil Compiler. For instructions on how to do so, see the Rigetti documentation `here <http://docs.rigetti.com/en/stable/start.html>`_.

In a Python3.6+ virtual environment you can install the `entropica_qaoa`  package using `pip <https://pip.pypa.io/en/stable/quickstart/>`_

.. code-block:: bash

   pip install entropica_qaoa

and if you have it already installed upgraded via

.. code-block:: bash

   pip install --upgrade entropica_qaoa

If you want to run the Demo Notebooks, you will additionally need `scikit-learn` and `scikit-optimize`, which can also be installed using pip:

.. code-block:: bash

   pip install scikit-learn && pip install scikit-optimize


First Steps
-----------

In :ref:`1-AnExampleWorkflow` we show you the basic usage of
EntropicaQAOA.
More advanced examples on working with different parametrizations are found in
:ref:`2-ParameterClasses`, and the more advanced features of the cost
functions are explained in :ref:`4-CostFunctionsAndVQE`. To learn how to conveniently
create QAOA problem instances, and convert between different formats, refer to :ref:`5-QAOAUtilities`.

The documentation can also downloaded as Jupyter notebooks from our `GitHub page <https://github.com/entropicalabs/entropica_qaoa/tree/master/examples>`_ .


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
