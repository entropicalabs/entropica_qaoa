QAOA Parametrizations
=====================

We offer (currently) 7 different parametrizations for QAOA that can be found in
the ``entropica_qaoa.qaoa.parameters`` module. They fall broadly into three categories: The `Standard` classes are parametrizations that have the
:math:`\gamma` 's and :math:`\beta` 's as free parameters as defined in the
seminal paper by Farhi et al. in
`A Quantum Approximate Optimization Algorithm
<https://arxiv.org/abs/1411.4028>`_.
The `Fourier` classes have the discrete cosine and sine transforms of the :math:`\gamma` 's respective :math:`\beta`'s as free parameters, as proposed by
Zhou et al. in
`Quantum Approximate Optimization Algorithm: Performance, Mechanism, and Implementation on Near-Term Devices
<https://arxiv.org/abs/1812.01041>`_.
The last class are the `Annealing` classes that are based on the idea of QAOA being discretized, adiabatci annealing. Here the function values :math:`s(t_i)`
at equally spaced times :math:`t_i` are the free parameters.

Except for the `Annealing` parameters these come also in three levels of detail: ``StandardParams`` and ``FourierParams`` offer the :math:`\gamma` 's and :math:`\beta` 's as proposed in above papers.  ``StandardWithBiasParams`` and ``FourierWithBiasParams`` allows for extra :math:`\gamma` 's for possible
one-qubit bias terms, resp. their discrete sine transform. Lastly,
``ExtendedParams`` and ``FourierExtendedParams`` offers full control by having a seperate set of rotation angles for each term in the cost and mixer hamiltonian, respective having a seperate set of fourier coefficients for each term.

You can always convert parametrisations with fewer degrees of freedom to ones with more using the ``.from_other_parameters()`` classmethod. The full type
tree is shown below and the arrows mark possible conversions:

.. code-block::

       ExtendedParams   <--------- FourierExtendedParams
              ^                         ^
              |                         |
    StandardWithBiasParams <------ FourierWithBiasParams
              ^                         ^
              |                         |
        StandardParams  <----------- FourierParams
              ^
              |
        AnnealingParams

Standard Parameters
-------------------

.. autoclass:: qaoa.parameters.StandardParams
    :members:
    :undoc-members:
    :inherited-members:


.. autoclass:: qaoa.parameters.StandardWithBiasParams
    :members:
    :undoc-members:
    :inherited-members:


.. autoclass:: qaoa.parameters.ExtendedParams
    :members:
    :undoc-members:
    :inherited-members:


Fourier Parameters
------------------

.. autoclass:: qaoa.parameters.FourierParams
    :members:
    :undoc-members:
    :inherited-members:


.. autoclass:: qaoa.parameters.FourierWithBiasParams
    :members:
    :undoc-members:
    :inherited-members:


.. autoclass:: qaoa.parameters.FourierExtendedParams
    :members:
    :undoc-members:
    :inherited-members:


Annealing Parameters
--------------------

.. autoclass:: qaoa.parameters.AnnealingParams
    :members:
    :undoc-members:
    :inherited-members:


Parameter Iterators
-------------------

To facilitate iterating one or more of the free parameters over a given range
(e.g. for landscape plots) we also provide a way to build Iterables that leave
all parameters except one fixed. The one is iterated over a range specified by the user.

.. autoclass:: qaoa.parameters.QAOAParameterIterator
    :members:
    :undoc-members:
