.. Entropica QAOA documentation master file, created by
   sphinx-quickstart on Tue Apr 23 00:13:37 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Entropica QAOA's documentation!
==========================================

Installation
------------

See ``README.md`` in the root directory of this package.


First Steps
-----------

You can find some Notebooks showing you the basic Usage of EntropiQAOA in the folder ``examples/``.


Compiling this Documentation
----------------------------

To compile this documentation you first must install Sphinx and some of its
extensions.
Sphinx and the neccesary extensions can be installed with

.. code-block:: bash

   pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints nbsphinx

(I hope this list is complete. Otherwise you will get errors during compilation
that hopefully tell you, what is missing).
Then the documentation can be compiled with

.. code-block:: bash

   cd docs && make html

(or however you use the ``make.bat`` on Windows)
and is found in ``docs/build/html``


Contents
--------

.. toctree::
   :maxdepth: 3
   :caption: The Demo Notebooks

   notebooks


.. toctree::
   :maxdepth: 1
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
