QAOA cost functions
===================

Like with VQE we also provide two different cost functions for QAOA that
derive from their VQE counterparts. One for the Wavefunction Simulator and one
for the QVM that can also be run on Rigettis QPU.

On the Wavefunction Simulator
-----------------------------

.. autoclass:: qaoa.cost_function.QAOACostFunctionOnWFSim
    :members:
    :undoc-members:
    :inherited-members:


On the QVM / QPU
----------------

.. autoclass:: qaoa.cost_function.QAOACostFunctionOnQVM
    :members:
    :undoc-members:
    :inherited-members:
