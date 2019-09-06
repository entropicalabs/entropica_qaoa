VQE cost functions
==================

We provide two different cost functions for VQE. One for PyQuil's WavefunctionSimulator, and one for Rigetti's QVM that can also be run on the QPU.

On the Wavefunction Simulator
-----------------------------

.. autoclass:: vqe.cost_function.PrepareAndMeasureOnWFSim
    :members:
    :undoc-members:
    :inherited-members:


On the QVM / QPU
----------------

.. autoclass:: vqe.cost_function.PrepareAndMeasureOnQVM
    :members:
    :undoc-members:
    :inherited-members:
