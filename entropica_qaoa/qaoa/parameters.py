#   Copyright 2019 Entropica Labs
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Different parametrizations of QAOA circuits.

This module holds an abstract class to store QAOA parameters in and (so far)
five derived classes that allow for more or less degrees of freedom in the QAOA
Ansatze

Todo
----
 - Better default values for ``time`` if ``None`` is passed
 - Better default parameters for ``fourier`` n_steps
 - implement ``AbstractParams.linear_ramp_from_hamiltonian()`` and
   then super() from it
"""

import copy
from typing import Iterable, Union, List, Tuple, Any, Type, Callable
import warnings
import math
from custom_inherit import DocInheritMeta

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, dst

from pyquil.paulis import PauliSum

# import entropica_qaoa.qaoa._parameter_conversions.converter as converter


def _is_iterable_empty(in_iterable):
    if isinstance(in_iterable, Iterable):    # Is Iterable
        return all(map(_is_iterable_empty, in_iterable))
    return False    # Not an Iterable


class shapedArray(object):
    """Decorator-Descriptor for arrays that have a fixed shape.

    This is used to facilitate automatic input checking for all the different
    internal parameters. Each instance of this class can be removed without
    replacement and the code should still work, provided the user provides
    only correct angles to below parameter classes

    Parameters
    ----------
    shape: Callable[[Any], Tuple]:
        Returns the shape for self.values

    Example
    -------
    With this descriptor, the two following are equivalent:

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)
                self._my_attribute = None

            @property
            def my_attribute(self):
                return _my_attribute

            @my_attribute.setter
            def my_attribute(self):
                try:
                    self._my_attribute = np.reshape(values, self.shape)
                except ValueError:
                    raise ValueError("my_attribute must have shape "
                                    f"{self.shape}")


    can be simplified to

    .. code-block:: python

        class foo():
            def __init__(self):
                self.shape = (n, m)

            @shapedArray
            def my_attribute(self):
                return self.shape
    """

    def __init__(self, shape: Callable[[Any], Tuple]):
        """The constructor. See class documentation"""
        self.name = shape.__name__
        self.shape = shape

    def __set__(self, obj, values):
        """The setter with input checking."""
        try:
            setattr(obj, f"__{self.name}", np.reshape(values, self.shape(obj)))
        except ValueError:
            raise ValueError(f"{self.name} must have shape {self.shape(obj)}")

    def __get__(self, obj, objtype):
        """The getter."""
        return getattr(obj, f"__{self.name}")


class AbstractParams(metaclass=DocInheritMeta(style="numpy")):
    """
    An abstract class to hold the parameters of a QAOA run and compute the
    angles from them.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian, the number of steps
        and possibly more (e.g. the total annealing time).
        ``hyperparametesr = (hamiltonian, n_steps, ...)``
    parameters: Tuple
        The QAOA parameters, that can be optimized. E.g. the gammas and betas
        or the annealing timesteps. AbstractParams doesn't implement
        this, but all child classes do.

    Attributes
    ----------
    pair_qubit_coeffs : np.array
        Coefficients of the pair terms in the hamiltonian
    qubits_pairs : List
        List of the qubit pairs in the hamiltonian. Same ordering as
        `pair_qubit_coeffs`
    single_qubit_coeffs : np.array
        Coefficients of the bias terms
    qubits_singles : List
        List of the bias qubits in the hamiltonian. Same ordering as
        `single_qubit_coeffs`
    reg : List
        List of all qubits for the X-rotations
    n_steps : int
        Number of timesteps. Corresponds to p_steps in the literature
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 hyperparameters: Tuple):
        # Note
        # ----
        # ``AbstractParams.__init__`` doesn't do anything with the
        # argument ``parameters``. In the child classes we have to super from
        # them and handle them correctly. Additionally we might need to handle
        # extra hyperparameters.

        hamiltonian, self.n_steps = hyperparameters[:2]

        # extract the qubit lists from the hamiltonian
        self.reg = hamiltonian.get_qubits()
        self.qubits_pairs = []
        self.qubits_singles = []

        # and fill qubits_singles and qubits_pairs according to the terms in the hamiltonian
        for term in hamiltonian:
            if len(term) == 1:
                self.qubits_singles.append(term.get_qubits()[0])
            elif len(term) == 2:
                self.qubits_pairs.append(term.get_qubits())
            elif len(term) == 0:
                pass  # could give a notice, that multiples of the identity are
                # ignored, since unphysical
            else:
                raise NotImplementedError(
                    "As of now we can only handle hamiltonians with at most"
                    " two-qubit terms")

        # extract the cofficients of the terms from the hamiltonian
        # if creating `ExtendedParams` form this, we can delete this attributes
        # again. Check if there are complex coefficients and issue a warning.
        self.single_qubit_coeffs = np.array([
            term.coefficient for term in hamiltonian if len(term) == 1])
        if np.any(np.iscomplex(self.single_qubit_coeffs)):
            warnings.warn("hamiltonian contained complex coefficients."
                          " Ignoring imaginary parts")
        self.single_qubit_coeffs = self.single_qubit_coeffs.real
        # and the same for the pair qubit coefficients
        self.pair_qubit_coeffs = np.array([
            term.coefficient for term in hamiltonian if len(term) == 2])
        if np.any(np.iscomplex(self.pair_qubit_coeffs)):
            warnings.warn("hamiltonian contained complex coefficients."
                          " Ignoring imaginary parts")
        self.pair_qubit_coeffs = self.pair_qubit_coeffs.real

    def __repr__(self):
        """Return an overview over the parameters and hyperparameters

        Todo
        ----
        Split this into ``__repr__`` and ``__str__`` with a more verbose
        output in ``__repr__``.
        """
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tsingle_qubit_coeffs: " + str(self.single_qubit_coeffs) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "\tpair_qubit_coeffs: " + str(self.pair_qubit_coeffs) + "\n"
        string += "\tn_steps: " + str(self.n_steps) + "\n"
        return string

    def __len__(self):
        """
        Returns
        -------
        int:
            the length of the data produced by self.raw() and accepted by
            self.update_from_raw()
        """
        raise NotImplementedError()

    @property
    def x_rotation_angles(self):
        """2D array with the X-rotation angles.

        1st index goes over n_steps and the 2nd index over the qubits to
        apply X-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError

    @property
    def z_rotation_angles(self):
        """2D array with the ZZ-rotation angles.

        1st index goes over the n_steps and the 2nd index over the qubit
        pairs, to apply ZZ-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError

    @property
    def zz_rotation_angles(self):
        """2D array with Z-rotation angles.

        1st index goes over the n_steps and the 2nd index over the qubit
        pairs, to apply Z-rotations on. These are needed by ``qaoa.cost_function.make_qaoa_memory_map``
        """
        raise NotImplementedError

    def update_from_raw(self, new_values: Union[list, np.array]):
        """
        Update all the parameters from a 1D array.

        The input has the same format as the output of ``self.raw()``.
        This is useful for ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Parameters
        ----------
        new_values:
            A 1D array with the new parameters. Must have length  ``len(self)``
            and the ordering of the flattend ``parameters`` in ``__init__()``.

        """
        raise NotImplementedError()

    def raw(self):
        """
        Return the parameters in a 1D array.

        This 1D array is needed by ``scipy.optimize.minimize`` which expects
        the parameters that need to be optimized to be a 1D array.

        Returns
        -------
        np.array:
            The parameters in a 1D array. Has the same output
            format as the expected input of ``self.update_from_raw``. Hence
            corresponds to the flattened `parameters` in `__init__()`

        """
        raise NotImplementedError()

    def raw_rotation_angles(self):
        """
        Flat array of the rotation angles for the memory map for the
        parametric circuit.

        Returns
        -------
        np.array:
            Returns all single rotation angles in the ordering
            ``(x_rotation_angles, gamma_singles, zz_rotation_angles)`` where
            ``x_rotation_angles = (beta_q0_t0, beta_q1_t0, ... , beta_qn_tp)``
            and the same for ``z_rotation_angles`` and ``zz_rotation_angles``

        """
        raw_data = np.concatenate((self.x_rotation_angles.flatten(),
                                   self.z_rotation_angles.flatten(),
                                   self.zz_rotation_angles.flatten()))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     time: float = None):
        """Alternative to ``__init__`` that already fills ``parameters``.

        Calculate initial parameters from a hamiltonian corresponding to a
        linear ramp annealing schedule and return a ``QAOAParams`` object.

        Parameters
        ----------
        hamiltonian:
            ``hamiltonian`` for which to calculate the initial QAOA parameters.
        n_steps:
            Number of timesteps.
        time:
            Total annealing time. Defaults to ``0.7*n_steps``.

        Returns
        -------
        Type[AbstractParams]
            The initial parameters for a linear ramp for ``hamiltonian``.

        """
        raise NotImplementedError()

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params,
                                parameter: Tuple =None):
        """Alternative to ``__init__`` that takes ``hyperparameters`` from an
        existing ``AbstractQAOAParameter`` object.

        Create a ConcreteQAOAParameters instance from an
        AbstractQAOAParameters instance with hyperparameters from
        ``abstract_params`` and normal parameters from ``parameters``


        Parameters
        ----------
        abstract_params: AbstractParams
            An AbstractQAOAParameters instance to which to add the parameters
        parameter:
            A tuple containing the parameters. Must be the same as in
            ``__init__``

        Returns
        -------
        AbstractParams
            A copy of itself. This function makes only sense for the child
            classes of ``AbstractQAOAParameters``
        """
        params = copy.deepcopy(abstract_params)
        params.__class__ = cls

        return params

    @classmethod
    def from_other_parameters(cls, params):
        """Alternative to ``__init__`` that takes parameters with less degrees
        of freedom as the input.

        Parameters
        ----------
        params: Type[AbstractParams]
            The input parameters

        Returns
        -------
        Type[AbstractParams]:
            The converted paramters s.t. all the rotation angles of the in
            and output parameters are the same.

        Note
        ----
        If all conversion functions for your parametrization are in
        qaoa._parameter_conversions.py and correctly registered for converter
        you don't need to override this function in its child classes
        """
        # Todo: Somehow fix this deferred import w/o creating cyclic imports?
        from entropica_qaoa.qaoa._parameter_conversions import converter
        return converter(params, cls)

    @classmethod
    def empty(cls, hyperparameters: tuple):
        """Alternative to ``__init__`` that only takes ``hyperparameters`` and
        fills ``parameters`` via ``np.empty``

        Parameters
        ----------
        hyperparameters:
            Same as the hyperparameters argument for ``cls.__init__()``

        Returns
        -------
        Type[AbstractParams]
            A Parameter object with the parameters filled by ``np.empty``
        """
        self = AbstractParams(hyperparameters)
        self.__class__ = cls
        return self

    def plot(self, ax=None, **kwargs):
        """
        Plots ``self`` in a sensible way to the canvas ``ax``, if provided.

        Parameters
        ----------
        ax: matplotlib.axes._subplots.AxesSubplot
            The canvas to plot itself on
        kwargs:
            All remaining keyword arguments are passed forward to the plot
            function

        """
        raise NotImplementedError()


class ExtendedParams(AbstractParams):
    r"""
    QAOA parameters in their most general form with different angles for each
    operator.

    This means, that at the i-th timestep the evolution hamiltonian is given by

    .. math::

        H(t_i) = \sum_{\textrm{qubits } j} \beta_{ij} X_j
               + \sum_{\textrm{qubits } j} \gamma_{\textrm{single } ij} Z_j
               + \sum_{\textrm{qubit pairs} (jk)} \gamma_{\textrm{pair } i(jk)} Z_j Z_k

    and the complete circuit is then

    .. math::

        U = e^{-i H(t_p)} \cdots e^{-iH(t_1)}.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian and the number of steps
        ``hyperparameters = (hamiltonian, n_steps)``
    parameters:
        Tuple containing ``(betas, gammas_singles, gammas_pairs)`` with dimensions
        ``((n_steps x nqubits), (n_steps x nsingle_terms), (n_steps x npair_terms))``

    Attributes
    ----------
    betas: np.array
        2D array with the X-rotation angles. 1st index goes over n_steps and
        the 2nd index over the qubits to apply X-rotations on.
    gammas_pairs: np.array
        2D array with the ZZ-rotation angles. 1st index goes over the n_steps
        and the 2nd index over the qubit pairs, to apply ZZ-rotations on.
    gammas_singles: np.array
        2D array with Z-rotation angles. 1st index goes over the n_steps
        and the 2nd index over the qubit pairs, to apply Z-rotations on.
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        # and add the parameters
        self.betas, self.gammas_singles, self.gammas_pairs\
            = np.array(parameters[0]), np.array(parameters[1]), np.array(parameters[2])

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tbetas: " + str(self.betas).replace("\n", ",") + "\n"
        string += "\tgammas_singles: " + str(self.gammas_singles)\
            .replace("\n", ",") + "\n"
        string += "\tgammas_pairs: " + str(self.gammas_pairs)\
            .replace("\n", ",") + "\n"
        return string

    def __len__(self):
        return self.n_steps * (len(self.reg) + len(self.qubits_pairs)
                               + len(self.qubits_singles))

    @shapedArray
    def betas(self):
        return (self.n_steps, len(self.reg))

    @shapedArray
    def gammas_singles(self):
        return (self.n_steps, len(self.qubits_singles))

    @shapedArray
    def gammas_pairs(self):
        return (self.n_steps, len(self.qubits_pairs))

    @property
    def x_rotation_angles(self):
        return self.betas

    @property
    def z_rotation_angles(self):
        return self.single_qubit_coeffs * self.gammas_singles

    @property
    def zz_rotation_angles(self):
        return self.pair_qubit_coeffs * self.gammas_pairs

    def update_from_raw(self, new_values):
        self.betas = np.array(new_values[:len(self.reg) * self.n_steps])
        self.betas = self.betas.reshape((self.n_steps, len(self.reg)))
        new_values = new_values[self.n_steps * len(self.reg):]

        self.gammas_singles = np.array(new_values[:len(self.qubits_singles)
                                                  * self.n_steps])
        self.gammas_singles = self.gammas_singles.reshape(
            (self.n_steps, len(self.qubits_singles)))
        new_values = new_values[self.n_steps * len(self.qubits_singles):]

        self.gammas_pairs = np.array(new_values[:len(self.qubits_pairs)
                                                * self.n_steps])
        self.gammas_pairs = self.gammas_pairs.reshape(
            (self.n_steps, len(self.qubits_pairs)))
        new_values = new_values[self.n_steps * len(self.qubits_pairs):]

        # PEP8 complains, but new_values could be np.array and not list!
        if not len(new_values) == 0:
            raise RuntimeWarning(
                "list to make new gammas and x_rotation_angles out of didn't"
                "have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.betas.flatten(),
                                   self.gammas_singles.flatten(),
                                   self.gammas_pairs.flatten()))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     time: float = None):
        """

        Returns
        -------
        ExtendedParams
            The initial parameters according to a linear ramp for
            ``hamiltonian``.

        Todo
        ----
        Refactor this s.t. it supers from __init__
        """
        # create evenly spaced timesteps at the centers of n_steps intervals
        if time is None:
            time = float(0.7 * n_steps)

        dt = time / n_steps
        times = np.linspace(time * (0.5 / n_steps), time
                            * (1 - 0.5 / n_steps), n_steps)

        term_lengths = [len(t) for t in hamiltonian]
        n_sing = term_lengths.count(1)
        n_pairs = term_lengths.count(2)
        n_betas = len(hamiltonian.get_qubits())
        has_higher_order_terms = any(l > 2 for l in term_lengths)
        if has_higher_order_terms:
            raise NotImplementedError(
                "As of now we can only handle hamiltonians with at most "
                "two-qubit terms")

        betas = np.array([dt * (1 - t / time) for t in times])
        betas = betas.repeat(n_betas).reshape(n_steps, n_betas)
        gammas_singles = np.array([dt * t / time for t in times])
        gammas_singles = gammas_singles.repeat(n_sing).reshape(n_steps, n_sing)
        gammas_pairs = np.array([dt * t / time for t in times])
        gammas_pairs = gammas_pairs.repeat(n_pairs).reshape(n_steps, n_pairs)

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps),
                     (betas, gammas_singles, gammas_pairs))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple) -> AbstractParams:
        """

        Returns
        -------
        ExtendedParams
            A ``GeneralQAOAParameters`` object with the hyperparameters taken from
            ``abstract_params`` and the normal parameters from ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        out.betas, out.gammas_singles, out.gammas_pairs\
            = np.array(parameters[0]), np.array(parameters[1]), np.array(parameters[2])
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        (self.betas, self.gammas_singles, self.gammas_pairs)\
            = (np.empty((self.n_steps, len(self.reg))),
               np.empty((self.n_steps, len(self.qubits_singles))),
               np.empty((self.n_steps, len(self.qubits_pairs))))
        return self

    def get_constraints(self):
        """Constraints on the parameters for constrained parameters.

        Returns
        -------
        List[Tuple]:
            A list of tuples (0, upper_boundary) of constraints on the
            parameters s.t. we are exploiting the periodicity of the cost
            function. Useful for constrained optimizers.

        """
        beta_constraints = [(0, 2 * math.pi)] * len(self.betas.flatten())
        gamma_pair_constraints = [(0, 2 * math.pi / w)
                                  for w in self.pair_qubit_coeffs]
        gamma_pair_constraints *= self.n_steps
        gamma_single_constraints = [(0, 2 * math.pi / w)
                                    for w in self.single_qubit_coeffs]
        gamma_single_constraints *= self.n_steps

        all_constraints = beta_constraints + gamma_single_constraints\
            + gamma_pair_constraints

        return all_constraints

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_singles):
            ax.plot(self.gammas_singles,
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_pairs):
            ax.plot(self.gammas_pairs,
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax.set_xlabel("timestep")
        ax.legend()


class StandardWithBiasParams(AbstractParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_{\textrm{singles}, p} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, p} H_{c, \textrm{pairs}}}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_{\textrm{singles}, 0} H_{c, \textrm{singles}}}
        e^{-i \gamma_{\textrm{pairs}, 0} H_{c, \textrm{pairs}}}

    where the cost hamiltonian is split into :math:`H_{c, \textrm{singles}}`
    the bias terms, that act on only one qubit, and
    :math:`H_{c, \textrm{pairs}}` the coupling terms, that act on two qubits.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian and the number of steps
        ``hyperparameters = (hamiltonian, n_steps)``
    parameters:
        Tuple containing ``(betas, gammas_singles, gammas_pairs)`` with
        dimensions ``(n_steps, n_steps, n_steps)``

    Attributes
    ----------
    betas: np.array
        A 1D array containing the betas from above for each timestep
    gammas_pairs: np.array
        A 1D array containing the gammas_singles from above for each timestep
    gammas_singles: np.array
        A 1D array containing the gammas_pairs from above for each timestep
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self.betas, self.gammas_singles, self.gammas_pairs\
            = np.array(parameters[0]), np.array(parameters[1]), np.array(parameters[2])

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas_singles: " + str(self.gammas_singles) + "\n"
        string += "\tgammas_pairs: " + str(self.gammas_pairs) + "\n"
        return(string)

    def __len__(self):
        return self.n_steps * 3

    @shapedArray
    def betas(self):
        return self.n_steps

    @shapedArray
    def gammas_singles(self):
        return self.n_steps

    @shapedArray
    def gammas_pairs(self):
        return self.n_steps

    @property
    def x_rotation_angles(self):
        return np.outer(self.betas, np.ones(len(self.reg)))

    @property
    def z_rotation_angles(self):
        return np.outer(self.gammas_singles, self.single_qubit_coeffs)

    @property
    def zz_rotation_angles(self):
        return np.outer(self.gammas_pairs, self.pair_qubit_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0:self.n_steps])
        new_values = new_values[self.n_steps:]    # cut betas from new_values
        self.gammas_singles = np.array(new_values[0:self.n_steps])
        new_values = new_values[self.n_steps:]
        self.gammas_pairs = np.array(new_values[0:self.n_steps])
        new_values = new_values[self.n_steps:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new gammas and x_rotation_angles out of"
                                 "didn't have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.betas,
                                   self.gammas_singles,
                                   self.gammas_pairs))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     time: float = None):
        """
        Returns
        -------
        StandardWithBiasParams
            An `AlternatingOperatorsQAOAParameters` object holding all the
            parameters
        """
        if time is None:
            time = float(0.7 * n_steps)
        # create evenly spaced n_steps at the centers of #n_steps intervals
        dt = time / n_steps
        times = np.linspace(time * (0.5 / n_steps), time
                            * (1 - 0.5 / n_steps), n_steps)

        # fill betas, gammas_singles and gammas_pairs
        # Todo (optional): replace by np.linspace for tiny performance gains
        betas = np.array([dt * (1 - t / time) for t in times])
        gammas_singles = np.array([dt * t / time for t in times])
        gammas_pairs = np.array([dt * t / time for t in times])

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps),
                     (betas, gammas_singles, gammas_pairs))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple):
        """

        Returns
        -------
        StandardWithBiasParams
            An ``AlternatatingOperatorsQAOAParameters`` object with the
            hyperparameters taken from ``abstract_params`` and the normal
            parameters from ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        out.betas, out.gammas_singles, out.gammas_pairs\
            = np.array(parameters[0]), np.array(parameters[1]), np.array(parameters[2])
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        (self.betas, self.gammas_singles, self.gammas_pairs)\
            = (np.empty((self.n_steps)),
               np.empty((self.n_steps)),
               np.empty((self.n_steps)))
        return self

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_singles):
            ax.plot(self.gammas_singles,
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.gammas_pairs):
            ax.plot(self.gammas_pairs,
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax.set_xlabel("timestep")
        # ax.grid(linestyle='--')
        ax.legend()


class StandardParams(AbstractParams):
    r"""
    QAOA parameters that implement a state preparation circuit with

    .. math::

        e^{-i \beta_p H_0}
        e^{-i \gamma_p H_c}
        \cdots
        e^{-i \beta_0 H_0}
        e^{-i \gamma_0 H_c}

    This corresponds to the parametrization used by Farhi in his
    original paper [https://arxiv.org/abs/1411.4028]

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian and the number of steps
        ``hyperparameters = (hamiltonian, n_steps)``
    parameters:
        Tuple containing ``(betas, gammas)`` with dimensions
        ``(n_steps, n_steps)``

    Attributes
    ----------
    betas: np.array
        1D array with the betas from above
    gammas: np.array
        1D array with the gamma from above
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self.betas, self.gammas\
            = np.array(parameters[0]), np.array(parameters[1])

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tbetas: " + str(self.betas) + "\n"
        string += "\tgammas: " + str(self.gammas) + "\n"
        return(string)

    def __len__(self):
        return self.n_steps * 2

    @shapedArray
    def betas(self):
        return self.n_steps

    @shapedArray
    def gammas(self):
        return self.n_steps

    @property
    def x_rotation_angles(self):
        return np.outer(self.betas, np.ones(len(self.reg)))

    @property
    def z_rotation_angles(self):
        return np.outer(self.gammas, self.single_qubit_coeffs)

    @property
    def zz_rotation_angles(self):
        return np.outer(self.gammas, self.pair_qubit_coeffs)

    def update_from_raw(self, new_values):
        # overwrite self.betas with new ones
        self.betas = np.array(new_values[0:self.n_steps])
        new_values = new_values[self.n_steps:]    # cut betas from new_values
        self.gammas = np.array(new_values[0:self.n_steps])
        new_values = new_values[self.n_steps:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new gammas and x_rotation_angles out of"
                                 "didn't have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.betas, self.gammas))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     time: float = None):
        """
        Returns
        -------
        StandardParams
            A ``StandardParams`` object with parameters according
            to a linear ramp schedule for ``hamiltonian``
        """
        if time is None:
            time = float(0.7 * n_steps)
        # create evenly spaced timesteps at the centers of n_steps intervals
        dt = time / n_steps
        times = np.linspace(time * (0.5 / n_steps), time
                            * (1 - 0.5 / n_steps), n_steps)

        # fill betas, gammas_singles and gammas_pairs
        # Todo (optional): replace by np.linspace for tiny performance gains
        betas = np.array([dt * (1 - t / time) for t in times])
        gammas = np.array([dt * t / time for t in times])

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps),
                     (betas, gammas))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple):
        """

        Returns
        -------
        StandardParams
            A ``StandardParams`` object with the hyperparameters
            taken from ``abstract_params`` and the normal parameters from
            ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        out.betas, out.gammas\
            = np.array(parameters[0]), np.array(parameters[1])
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        (self.betas, self.gammas) = (np.empty((self.n_steps)),
                                     np.empty((self.n_steps)))
        return self

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="", **kwargs)
        ax.plot(self.gammas, label="gammas", marker="^", ls="", **kwargs)
        ax.set_xlabel("timestep", fontsize=12)
        ax.legend()


class AnnealingParams(AbstractParams):
    """
    QAOA parameters that implement a state preparation circuit of the form

    .. math::

        U = e^{-i (1-s(t_p)) H_M \Delta t} e^{-i s(t_p) H_C \Delta t} \cdots e^{-i(1-s(t_1)H_M \Delta t} e^{-i s(t_1) H_C \Delta t}

    where the :math:`s(t_i) =: s_i` are the variable parameters and
    :math:`\\Delta t= \\frac{T}{N}`.
    So the schedule is specified by specifying s(t) at evenly spaced timesteps.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian, the number of steps
        and the total annealing time ``hyperparameters = (hamiltonian,
        n_steps, time)``
    parameters : Tuple
        Tuple containing ``(schedule values)`` of length ``n_steps``

    Attributes
    ----------
    schedule: np.array
        An 1D array holding the values of the schedule function at each timestep.
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int, float],
                 parameters: List):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self._annealing_time = hyperparameters[2]
        self.schedule = np.array(parameters)

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tschedule: " + str(self.schedule)
        return string

    def __len__(self):
        return self.n_steps

    @shapedArray
    def schedule(self):
        return self.n_steps

    @property
    def x_rotation_angles(self):
        dt = self._annealing_time / self.n_steps
        tmp = (1 - self.schedule) * dt
        return np.outer(tmp, np.ones(len(self.reg)))

    @property
    def z_rotation_angles(self):
        dt = self._annealing_time / self.n_steps
        tmp = self.schedule * dt
        return np.outer(tmp, self.single_qubit_coeffs)

    @property
    def zz_rotation_angles(self):
        dt = self._annealing_time / self.n_steps
        tmp = self.schedule * dt
        return np.outer(tmp, self.pair_qubit_coeffs)

    def update_from_raw(self, new_values):
        if len(new_values) != self.n_steps:
            raise RuntimeWarning(
                "the new times should have length n_steps+1")
        self.schedule = np.array(new_values)

    def raw(self):
        return self.schedule

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     time: float = None):
        """
        Returns
        -------
        AnnealingParams :
            An ``AnnealingParams`` object corresponding to
            a linear ramp schedule.
        """
        if time is None:
            time = 0.7 * n_steps

        schedule = np.linspace(0.5 / n_steps, 1 - 0.5 / n_steps, n_steps)

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps, time), (schedule))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple,
                                time: float = None):
        """

        Returns
        -------
        AnnealingParams
            An ``AnnealingParams`` object with the hyperparameters taken from
            ``abstract_params`` and the normal parameters from ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        if time is None:
            time = 0.7 * out.n_steps
        out._annealing_time = time
        out.schedule = np.array(parameters)
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        self._annealing_time = hyperparameters[2]
        self.schedule = np.empty((self.n_steps))
        return self

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.schedule, marker="s", **kwargs)
        ax.set_xlabel("timestep number", fontsize=14)
        ax.set_ylabel("s(t)", fontsize=14)


class FourierParams(AbstractParams):
    """
    The QAOA parameters as the sine/cosine transform of the original gammas
    and x_rotation_angles. See "Quantum Approximate Optimization Algorithm:
    Performance, Mechanism, and Implementation on Near-Term Devices"
    [https://arxiv.org/abs/1812.01041] for a detailed description.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian, the number of steps
        and the total annealing time
        ``hyperparameters = (hamiltonian, n_steps, q)``.
        ``q`` is the number of fourier coefficients. For ``q == n_steps`` we
        have the full expressivity of ``StandardParams``.
        More are redundant.
    parameters:
        Tuple containing ``(v, u)`` with dimensions
        ``(q, q)``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    u : np.array
        The discrete sine transform of the ``gammas`` in
        ``StandardParams``
    v : np.array
        The discrete cosine transform of the ``betas`` in
        ``StandardParams``
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int, float],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self.q = hyperparameters[2]
        self.v, self.u = parameters

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles" + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs" + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tv: " + str(self.v) + "\n"
        string += "\tu: " + str(self.u) + "\n"
        return(string)

    def __len__(self):
        return 2 * self.q

    @shapedArray
    def v(self):
        return self.q

    @shapedArray
    def u(self):
        return self.q

    @property
    def x_rotation_angles(self):
        betas = dct(self.v, n=self.n_steps)
        return np.outer(betas, np.ones(len(self.reg)))

    @property
    def z_rotation_angles(self):
        gammas = dst(self.u, n=self.n_steps)
        return np.outer(gammas, self.single_qubit_coeffs)

    @property
    def zz_rotation_angles(self):
        gammas = dst(self.u, n=self.n_steps)
        return np.outer(gammas, self.pair_qubit_coeffs)

    def update_from_raw(self, new_values):
        # overwrite x_rotation_angles with new ones
        self.v = np.array(new_values[0:self.q])
        # cut x_rotation_angles from new_values
        new_values = new_values[self.q:]
        self.u = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new u's and v's out of"
                                 "didn't have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.v, self.u))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     q: int = 4,
                                     time: float = None):
        """
        Parameters
        ----------
        hamiltonian:
            The hamiltonian
        n_steps:
            number of timesteps
        q:
            Number of Fourier coefficients. Defaults to 4
        time:
            total time. Set to ``0.7*n_steps`` if ``None`` is passed.

        Returns
        -------
        FourierParams:
            A ``FourierParams`` object with initial parameters
            corresponding to a linear ramp annealing schedule

        ToDo
        ----
        Make a more informed choice of the default value for ``q``. Probably
        depending on ``n_qubits``
        """
        if time is None:
            time = 0.7 * n_steps

        # fill x_rotation_angles, z_rotation_angles and zz_rotation_angles
        # Todo make this an easier expresssion
        v = np.zeros(q)
        v[0] = 0.5 * time / n_steps
        u = np.copy(v)

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps, q), (v, u))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple,
                                q: int = 4):
        """

        Parameters
        ----------
        abstract_params:
            An AbstractQAOAParameters instance to which to add the parameters
        parameters:
            Same as ``parameters`` in ``.__init__()``
        q:
            Number of fourier coefficients. Defaults to 4

        Returns
        -------
        FourierParams
            A ``FourierParams`` object with the hyperparameters taken
            from ``abstract_params`` and the normal parameters from
            ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        out.q = q
        out.v, out.u =\
            np.array(parameters[0]), np.array(parameters[1])
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        self.q = hyperparameters[2]
        self.v, self.u = np.empty((self.q)), np.empty((self.q))
        return self

    def plot(self, ax=None, **kwargs):
        warnings.warn("Plotting the gammas and x_rotation_angles through DCT "
                      "and DST. If you are interested in v, u you can access "
                      "them via params.v, params.u")
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(dct(self.v, n=self.n_steps),
                label="betas", marker="s", ls="", **kwargs)
        ax.plot(dst(self.u, n=self.n_steps),
                label="gammas", marker="v", ls="", **kwargs)
        ax.set_xlabel("timestep")
        ax.legend()


class FourierWithBiasParams(AbstractParams):
    """
    The QAOA parameters as the sine/cosine transform of the original gammas
    and x_rotation_angles. See "Quantum Approximate Optimization Algorithm:
    Performance, Mechanism, and Implementation on Near-Term Devices"
    [https://arxiv.org/abs/1812.01041] for a detailed description.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian, the number of steps
        and the total annealing time
        ``hyperparameters = (hamiltonian, n_steps, q)``.
        ``q`` is the number of fourier coefficients. For ``q == n_steps`` we
        have the full expressivity of ``StandardWithBiasParams``.
        More are redundant.
    parameters:
        Tuple containing ``(v, u_singles, u_pairs)`` with dimensions
        ``(q, q, q)``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    u_pairs : np.array
        The discrete sine transform of the ``gammas_pairs`` in
        ``StandardWithBiasParams``
    u_singles : np.array
        The discrete sine transform of the ``gammas_singles`` in
        ``StandardWithBiasParams``
    v : np.array
        The discrete cosine transform of the betas in
        ``StandardWithBiasParams``
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int, float],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self.q = hyperparameters[2]
        self.v, self.u_singles, self.u_pairs = parameters

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "Parameters:\n"
        string += "\tu_singles: " + str(self.u_singles) + "\n"
        string += "\tu_pairs: " + str(self.u_pairs) + "\n"
        string += "\tv: " + str(self.v) + "\n"
        return(string)

    def __len__(self):
        return 3 * self.q

    @shapedArray
    def v(self):
        return self.q

    @shapedArray
    def u_singles(self):
        return self.q

    @shapedArray
    def u_pairs(self):
        return self.q

    @property
    def x_rotation_angles(self):
        betas = dct(self.v, n=self.n_steps)
        return np.outer(betas, np.ones(len(self.reg)))

    @property
    def z_rotation_angles(self):
        gammas_singles = dst(self.u_singles, n=self.n_steps)
        return np.outer(gammas_singles, self.single_qubit_coeffs)

    @property
    def zz_rotation_angles(self):
        gammas_pairs = dst(self.u_pairs, n=self.n_steps)
        return np.outer(gammas_pairs, self.pair_qubit_coeffs)

    def update_from_raw(self, new_values):
        # overwrite x_rotation_angles with new ones
        self.v = np.array(new_values[0:self.q])
        # cut x_rotation_angles from new_values
        new_values = new_values[self.q:]
        self.u_singles = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]
        self.u_pairs = np.array(new_values[0:self.q])
        new_values = new_values[self.q:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new u's and v's out of"
                                 "didn't have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.v,
                                   self.u_singles,
                                   self.u_pairs))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     q: int = 4,
                                     time: float = None):
        """
        Parameters
        ----------
        hamiltonian:
            The hamiltonian
        n_steps:
            number of timesteps
        q:
            Number of Fourier coefficients. Defaults to 4
        time:
            total time. Set to ``0.7*n_steps`` if ``None`` is passed.

        Returns
        -------
        FourierWithBiasParams:
            A ``FourierWithBiasParams`` object with initial parameters
            corresponding to a linear ramp annealing schedule

        ToDo
        ----
        Make a more informed choice of the default value for ``q``. Probably
        depending on ``n_qubits``
        """
        if time is None:
            time = 0.7 * n_steps

        v = np.zeros(q)
        v[0] = 0.5 * time / n_steps
        u_singles = np.copy(v)
        u_pairs = np.copy(v)
        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps, q), (v, u_singles, u_pairs))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple,
                                q: int = 4):
        """

        Parameters
        ----------
        abstract_params:
            An AbstractQAOAParameters instance to which to add the parameters
        parameters:
            Same as ``parameters`` in ``.__init__()``
        q:
            Number of fourier coefficients. Defaults to 4

        Returns
        -------
        FourierWithBiasParams
            A ``FourierWithBiasParams`` object with the hyperparameters taken
            from ``abstract_params`` and the normal parameters from
            ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        if q is None:
            q = 4
        out.q = q
        out.v, out.u_singles, out.u_pairs =\
            np.array(parameters[0]), np.array(parameters[1]), np.array(parameters[2])
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        self.q = hyperparameters[2]
        self.v, self.u_singles, self.u_pairs\
            = np.empty((self.q)), np.empty((self.q)), np.empty((self.q))
        return self

    def plot(self, ax=None, **kwargs):
        warnings.warn("Plotting the gammas and x_rotation_angles through DCT "
                      "and DST. If you are interested in v, u_singles and "
                      "u_pairs you can access them via params.v, "
                      "params.u_singles, params.u_pairs")
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(dct(self.v, n=self.n_steps),
                label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.u_singles):
            ax.plot(dst(self.u_singles, n=self.n_steps),
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.u_pairs):
            ax.plot(dst(self.u_pairs, n=self.n_steps),
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax.set_xlabel("timestep")
        ax.legend()


class FourierExtendedParams(AbstractParams):
    r"""
    The Fourier pendant to ExtendedParams.

    Parameters
    ----------
    hyperparameters:
        The hyperparameters containing the hamiltonian and the number of steps
        and the number of fourier coefficients
        ``hyperparameters = (hamiltonian, n_steps, q)``
    parameters:
        Tuple containing ``(v, u_singles, u_pairs)`` with dimensions
        ``((q x nqubits), (q x nsingle_terms), (q x npair_terms))``

    Attributes
    ----------
    q : int
        The number of coefficients for the discrete sine and cosine transforms
        below
    v: np.array
        The discrete cosine transform of the ``betas`` in ``ExtendedParams``
    u_singles: np.array
        The discrete sine transform of the ``gammas_singles`` in
        ``ExtendedParams``
    u_pairs: np.array
        The discrete sine transform of the ``gammas_pairs`` in
        ``ExtendedParams``
    """

    def __init__(self,
                 hyperparameters: Tuple[PauliSum, int],
                 parameters: Tuple):
        # setup reg, qubits_singles and qubits_pairs
        super().__init__(hyperparameters)
        self.q = hyperparameters[2]
        self.v, self.u_singles, self.u_pairs = (np.array(parameters[0]),
                                                np.array(parameters[1]),
                                                np.array(parameters[2]))

    def __repr__(self):
        string = "Hyperparameters:\n"
        string += "\tregister: " + str(self.reg) + "\n"
        string += "\tqubits_singles: " + str(self.qubits_singles) + "\n"
        string += "\tqubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "Parameters:\n"
        string += "\tv: " + str(self.v).replace("\n", ",") + "\n"
        string += "\tu_singles: " + str(self.u_singles)\
            .replace("\n", ",") + "\n"
        string += "\tu_pairs: " + str(self.u_pairs)\
            .replace("\n", ",") + "\n"
        return string

    def __len__(self):
        return self.q * (len(self.reg) + len(self.qubits_pairs)
                         + len(self.qubits_singles))

    @shapedArray
    def v(self):
        return (self.q, len(self.reg))

    @shapedArray
    def u_singles(self):
        return (self.q, len(self.qubits_singles))

    @shapedArray
    def u_pairs(self):
        return (self.q, len(self.qubits_pairs))

    @property
    def x_rotation_angles(self):
        betas = dct(self.v, n=self.n_steps, axis=0)
        return betas

    @property
    def z_rotation_angles(self):
        gammas_singles = dst(self.u_singles,
                             n=self.n_steps, axis=0)
        return self.single_qubit_coeffs * gammas_singles

    @property
    def zz_rotation_angles(self):
        gammas_pairs = dst(self.u_pairs,
                           n=self.n_steps, axis=0)
        return self.pair_qubit_coeffs * gammas_pairs

    def update_from_raw(self, new_values):
        self.v = np.array(new_values[:len(self.reg) * self.q])
        self.v = self.v.reshape((self.q, len(self.reg)))
        new_values = new_values[self.q * len(self.reg):]

        self.u_singles = np.array(new_values[:len(self.qubits_singles)
                                             * self.q])
        self.u_singles = self.u_singles.reshape((self.q,
                                                 len(self.qubits_singles)))
        new_values = new_values[self.q * len(self.qubits_singles):]

        self.u_pairs = np.array(new_values[:len(self.qubits_pairs)
                                           * self.q])
        self.u_pairs = self.u_pairs.reshape((self.q,
                                             len(self.qubits_pairs)))
        new_values = new_values[self.q * len(self.qubits_pairs):]

        # PEP8 complains, but new_values could be np.array and not list!
        if not len(new_values) == 0:
            raise RuntimeWarning(
                "list to make new u's and v's out of didn't"
                "have the right length!")

    def raw(self):
        raw_data = np.concatenate((self.v.flatten(),
                                   self.u_singles.flatten(),
                                   self.u_pairs.flatten()))
        return raw_data

    @classmethod
    def linear_ramp_from_hamiltonian(cls,
                                     hamiltonian: PauliSum,
                                     n_steps: int,
                                     q: int = 4,
                                     time: float = None):
        """
        Returns
        -------
        FourierExtendedParams
            The initial parameters according to a linear ramp for
            ``hamiltonian``.

        """
        # create evenly spaced timesteps at the centers of n_steps intervals
        if time is None:
            time = float(0.7 * n_steps)

        term_lengths = [len(t) for t in hamiltonian]
        n_sing = term_lengths.count(1)
        n_pairs = term_lengths.count(2)
        n_betas = len(hamiltonian.get_qubits())
        has_higher_order_terms = any(l > 2 for l in term_lengths)
        if has_higher_order_terms:
            raise NotImplementedError(
                "As of now we can only handle hamiltonians with at most "
                "two-qubit terms")

        v = np.zeros(q)
        v[0] = 0.5 * time / n_steps
        u_singles = np.copy(v)
        u_pairs = np.copy(v)

        v = v.repeat(n_betas).reshape(q, n_betas)
        u_singles = u_singles.repeat(n_sing).reshape(q, n_sing)
        u_pairs = u_pairs.repeat(n_pairs).reshape(q, n_pairs)

        # wrap it all nicely in a qaoa_parameters object
        params = cls((hamiltonian, n_steps, q), (v, u_singles, u_pairs))
        return params

    @classmethod
    def from_AbstractParameters(cls,
                                abstract_params: AbstractParams,
                                parameters: Tuple,
                                q: int = 4) -> AbstractParams:
        """

        Returns
        -------
        FourierExtendedParams
            A ``FourierExtendedParams`` object with the hyperparameters taken
            from ``abstract_params`` and the normal parameters from
            ``parameters``
        """
        out = super().from_AbstractParameters(abstract_params)
        if q is None:
            q = 4
        out.q = q
        out.v, out.u_singles, out.u_pairs = (np.array(parameters[0]),
                                             np.array(parameters[1]),
                                             np.array(parameters[2]))
        return out

    @classmethod
    def empty(cls, hyperparameters):
        self = super().empty(hyperparameters)
        self.q = hyperparameters[2]
        (self.v, self.u_singles, self.u_pairs)\
            = (np.empty((self.q, len(self.reg))),
               np.empty((self.q, len(self.qubits_singles))),
               np.empty((self.q, len(self.qubits_pairs))))
        return self

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(dct(self.v, n=self.n_steps, axis=0),
                label="betas", marker="s", ls="", **kwargs)
        if not _is_iterable_empty(self.u_singles):
            ax.plot(dst(self.u_singles, n=self.n_steps),
                    label="gammas_singles", marker="^", ls="", **kwargs)
        if not _is_iterable_empty(self.u_pairs):
            ax.plot(dst(self.u_pairs, n=self.n_steps),
                    label="gammas_pairs", marker="v", ls="", **kwargs)
        ax.set_xlabel("timestep")
        ax.legend()


class QAOAParameterIterator:
    """An iterator to sweep one parameter over a range in a QAOAParameter object.

    Parameters
    ----------
    qaoa_params:
        The initial QAOA parameters, where one of them is swept over
    the_parameter:
        A string specifying, which parameter should be varied. It has to be
        of the form ``<attr_name>[i]`` where ``<attr_name>`` is the name
        of the _internal_ list and ``i`` the index, at which it sits. E.g.
        if ``qaoa_params`` is of type ``AnnealingParams``
        and  we want to vary over the second timestep, it is
        ``the_parameter = "times[1]"``.
    the_range:
        The range, that ``the_parameter`` should be varied over

    Todo
    ----
    - Add checks, that the number of indices in ``the_parameter`` matches
      the dimensions of ``the_parameter``
    - Add checks, that the index is not too large

    Example
    -------
    Assume qaoa_params is of type ``StandardWithBiasParams`` and
    has `n_steps >= 2`. Then the following code produces a loop that
    sweeps ``gammas_singles[1]`` over the range ``(0, 1)`` in 4 steps:

    .. code-block:: python

        the_range = np.arange(0, 1, 0.4)
        the_parameter = "gammas_singles[1]"
        param_iterator = QAOAParameterIterator(qaoa_params, the_parameter, the_range)
        for params in param_iterator:
            # do what ever needs to be done.
            # we have type(params) == type(qaoa_params)
    """

    def __init__(self, qaoa_params: Type[AbstractParams],
                 the_parameter: str,
                 the_range: Iterable[float]):
        """See class documentation for details"""
        self.params = qaoa_params
        self.iterator = iter(the_range)
        self.the_parameter, *indices = the_parameter.split("[")
        indices = [i.replace(']', '') for i in indices]
        if len(indices) == 1:
            self.index0 = int(indices[0])
            self.index1 = False
        elif len(indices) == 2:
            self.index0 = int(indices[0])
            self.index1 = int(indices[1])
        else:
            raise ValueError("the_parameter has to many indices")

    def __iter__(self):
        return self

    def __next__(self):
        # get next value from the_range
        value = next(self.iterator)

        # 2d list or 1d list?
        if self.index1 is not False:
            getattr(self.params, self.the_parameter)[self.index0][self.index1] = value
        else:
            getattr(self.params, self.the_parameter)[self.index0] = value

        return self.params
