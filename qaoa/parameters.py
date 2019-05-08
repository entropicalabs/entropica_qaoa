"""Different parametrizations of QAOA circuits.

This module holds an abstract class to store QAOA parameters in and (so far)
four derived classes that allow for more or less degrees of freedom in the QAOA
Ansätze

Todo
----
 - Better default values for ``time`` if ``None`` is passed
 - Better default parameters for ``fourier`` timesteps
 - implement AbstractQAOAParameters.from_hamiltonian() and then super() from it
"""

from typing import Iterable, Union, List, Tuple, Any
import warnings

# from custom_inherit import DocInheritMeta
import matplotlib.pyplot as plt
import numpy as np

from pyquil.paulis import PauliSum


def _is_list_empty(in_list):
    if isinstance(in_list, list):    # Is a list
        return all(map(_is_list_empty, in_list))
    return False    # Not a list


class AbstractQAOAParameters():
    """
    An abstract class to hold the parameters of a QAOA
    run and compute the angles from them.

    Parameters
    ----------
    constant_parameters : Any
        The constant parameters like the hamiltonian or number of steps.
        More in set_constant_parameters()
    variable_parameters : Any
        The variable parameters like the angles or stepwidths. More details in
        update_variable_parameters()

    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 constant_parameters: Tuple[Any, Iterable, Iterable, Iterable],
                 variable_parameters: Tuple = None):
        """
        Sets all the constant parameters via
        ``self.set_constant_parameters()`` and variable parameters via
        ``self.update_variable_parameters``
        """
        # This is just to shut the linter up. They are actually set
        # in self.update_variable_parameters
        self.betas = []
        self.gammas_singles = []
        self.gammas_pairs = []

        self.set_constant_parameters(constant_parameters)
        if variable_parameters is not None:
            self.update_variable_parameters(variable_parameters)

    def __repr__(self):
        raise NotImplementedError()

    def __len__(self):
        """
        Returns
        -------
        int:
            the length of the data produced by self.raw() and accepted by
            self.update()
        """
        raise NotImplementedError()

    def set_constant_parameters(self, constant_parameters):
        """
        Sets the constant parameters, like the register for the x-rotations and
        qubit lists for the single and double qubit z-rotation.

        Must initialize
        - `self.reg` the register to apply the x rotations on
        - `self.qubits_singles` the list of qubits to apply the single qubit
           z rotations on
        - `self.qubits_pairs` the list of qubits to apply the double qubit
          z rotations on
        - `self.timesteps` the number of timesteps
        """
        raise NotImplementedError()

    def update_variable_parameters(self, variable_parameters: Tuple):
        """
        Updates the variable parameters (i.e. angle parameters) s.t.
 
        - ``self.betas`` is a list/array of the x-rotation angles.
            Must have `dim self.timesteps` x ``len(self.reg)``
        - ``self.gammas_singles`` is the list of the single qubit Z-rotation angles.
            Must have dim ``dim self.timesteps`` x ``len(self.qubits_singles)``
        - ``self.gammas_pairs`` is the list of the two qubit Z-rotation angles.
            Must have dim ``dim self.timesteps`` x ``len(self.qubits_pairs)``
        """
        raise NotImplementedError()

    def update(self, new_values: Union[list, np.array]):
        """
        Updates all the angles based on a 1D array whose shape is specified later.
        The input has the same format as the output of ``self.raw()``.
        This is useful for ``scipy.optimize.minimize`` which only minimizes
        w.r.t a 1D array of parameters

        Parameters
        ----------
        new_values : Union[list, np.array]
            A 1D array with the new parameters. Must have length  ``len(self)``

        """
        raise NotImplementedError()

    def raw(self):
        """
        Returns the angles in a 1D array. This is needed by ``scipy.optimize.minimize``
        which only minimizes w.r.t a 1D array of parameters

        Returns
        -------
        np.array :
            all the tunable parameters in a 1D array. Has the same output
            format as the expected input of ``self.update``

        """
        raise NotImplementedError()

    def raw_all(self):
        """
        Returns all single rotation angles as needed for the memory map in parametric circuits

        Returns
        -------
        Union[List, np.array] :
            Returns all single rotation angles in the ordering
            (betas, gamma_singles, gammas_pairs) where
            betas = (beta_q0_t0, beta_q1_t0, ... , beta_qn_tp)
            and the same for gammas_singles and gammas_pairs

        """
        raw_data = []
        raw_data += [beta for betas in self.betas for beta in betas]
        raw_data += [g for gammas in self.gammas_singles for g in gammas]
        raw_data += [g for gammas in self.gammas_pairs for g in gammas]
        return raw_data

    @classmethod
    def from_hamiltonian(cls,
                         cost_hamiltonian: PauliSum,
                         timesteps: int,
                         time: float = None,
                         reg: List = None):
        """
        Calculate initial parameters from a hamiltonian corresponding to a
        linear ramp annealing schedule.

        Parameters
        ----------
        cost_hamiltonian : PauliSum
            `cost_hamiltonian` for which to calcuate the initial QAOA parameters.
        reg : List
            (Optional) Register of qubits on which `cost_hamiltonian` acts.
            If None is passed, cost_hamiltonian.get_qubits() is used.
        timesteps : int
            Number of timesteps.
        time : float
            Total annealing time. If none is passed, 0,7 * timesteps is used.

        Returns
        -------
        AbstractQAOAParameters
            The initial parameters best for `cost_hamiltonian`.

        """
        raise NotImplementedError()

    # TODO pass kwargs forward to plot() in all implementations  of this.
    def plot(self, ax=None):
        """
        Plots ``self`` in a sensible way to the canvas ``ax``, if provided.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.AxesSubplot
            The canvas to plot itself on

        """
        raise NotImplementedError()


class GeneralQAOAParameters(AbstractQAOAParameters):
    """
    QAOA parameters in their most general form with different angles for each
    operator.

    Todo
    ----
    Put a nice equation like U = exp(-i*gamma_{00}) * exp(...) here to explain
    better what we mean with most general?
    """

    def __repr__(self):
        string = "register: " + str(self.reg) + "\n"
        string += "betas: " + str(self.betas) + "\n"
        string += "qubits_singles: " + str(self.qubits_singles) + "\n"
        string += "gammas_singles: " + str(self.gammas_singles) + "\n"
        string += "qubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "gammas_pairs: " + str(self.gammas_pairs) + "\n"
        return string

    def __len__(self):
        return self.timesteps * (len(self.reg) + len(self.qubits_pairs)
                                 + len(self.qubits_singles))

    def set_constant_parameters(self,
                                constant_parameters: Tuple):
        """
        Parameters
        ----------
        constant_parameters :  Tuple
            a tuple containing ``(reg, qubits_singles, qubits_pairs, timesteps)``
            of types (List, Union[List, np,array], Union[List, np,array],
                      Union[List, np,array])
        """
        self.reg, self.qubits_singles, self.qubits_pairs, self.timesteps\
            = constant_parameters

    def update_variable_parameters(self, variable_parameters: Tuple[Union[List, np.array]]):
        """
        Parameters
        ----------
        variable_parameters:  Tuple
            a tuple containing ``(betas, gammas_singles, gammas_pairs)``
            in their fully extended form.
        """
        self.betas, self.gammas_singles, self.gammas_pairs = variable_parameters

        # and check that the data makes sense...
        if (self.timesteps != len(self.betas)
                or len(self.betas[0]) != len(self.reg)):
            raise ValueError("Check the dimensions of betas")

        if (not _is_list_empty(self.gammas_singles)
                and (self.timesteps != len(self.gammas_singles)
                     or len(self.gammas_singles[0]) != len(self.qubits_singles))):
            raise ValueError("Check the dimensions of gammas_singles")

        if (not _is_list_empty(self.gammas_pairs)
                and (self.timesteps != len(self.gammas_pairs)
                     or len(self.gammas_pairs[0]) != len(self.qubits_pairs))):
            raise ValueError("Check the dimensions of gammas_pairs")

    def update(self, new_values):
        self.betas = [new_values[len(self.reg) * i:len(self.reg) * i + len(self.reg)]
                      for i in range(self.timesteps)]
        new_values = new_values[self.timesteps * len(self.reg):]

        self.gammas_singles =\
            [new_values[len(self.qubits_singles) * i:len(self.qubits_singles) * i
                        + len(self.qubits_singles)] for i in range(self.timesteps)]
        new_values = new_values[self.timesteps * len(self.qubits_singles):]

        self.gammas_pairs =\
            [new_values[len(self.qubits_pairs) * i:len(self.qubits_pairs) * i
                        + len(self.qubits_pairs)] for i in range(self.timesteps)]
        new_values = new_values[self.timesteps * len(self.qubits_pairs):]

        # PEP8 complains, but new_values could be np.array and not list!
        if  not len(new_values) == 0:
            raise RuntimeWarning(
                "list to make new gammas and betas out of didn't have the right length!")

    def raw(self):
        raw_data = []
        raw_data += [beta for betas in self.betas for beta in betas]
        raw_data += [g for gammas in self.gammas_singles for g in gammas]
        raw_data += [g for gammas in self.gammas_pairs for g in gammas]
        return raw_data

    @classmethod
    def from_hamiltonian(cls,
                         cost_hamiltonian: PauliSum,
                         timesteps: int,
                         time: float = None,
                         reg: List = None):
        """
        Calculate initial parameters from a hamiltonian corresponding to a
        linear ramp annealing schedule.

        Parameters
        ----------
        cost_hamiltonian : PauliSum
            `cost_hamiltonian` for which to calcuate the initial QAOA parameters.
        reg : List
            (Optional) Register of qubits on which `cost_hamiltonian` acts.
            If None is passed, cost_hamiltonian.get_qubits() is used.
        timesteps : int
            Number of timesteps.
        time : float
            Total annealing time. If none is passed, 0,7 * timesteps is used.

        Returns
        -------
        AbstractQAOAParameters
            The initial parameters best for `cost_hamiltonian`.

        """
        # create evenly spaced timesteps at the centers of #timesteps intervals
        if time is None:
            time = float(0.7 * timesteps)

        if reg is None:
            reg = cost_hamiltonian.get_qubits()

        dt = time / timesteps
        times = np.linspace(time * (0.5 / timesteps), time
                            * (1 - 0.5 / timesteps), timesteps)

        betas = []
        gammas_pairs = []
        qubits_pairs = []
        gammas_singles = []
        qubits_singles = []

        # fill qubits_singles and qubits_pairs according to the terms in the hamiltonian
        for term in cost_hamiltonian:
            if len(term) == 1:
                # needs fixing...
                qubits_singles.append(term.get_qubits()[0])
            elif len(term) == 2:
                qubits_pairs.append(term.get_qubits())
            elif len(term) == 0:
                pass  # could give a notice, that multiples of the identity are
                      # ignored, since unphysical
            else:
                raise NotImplementedError(
                    "As of now we can only handle hamiltonians with at most two-qubit terms")

        # fill gammas_singles and gammas_pairs according to the timesteps and
        # coefficients of the terms in the hamiltonian
        for t in times:
            gamma_pairs = []
            gamma_singles = []
            beta = []

            for term in cost_hamiltonian:
                if len(term) == 1:
                    gamma_singles.append(t * term.coefficient.real * dt / time)

                elif len(term) == 2:
                    gamma_pairs.append(t * term.coefficient.real * dt / time)
                elif len(term) == 0:
                    pass  # could give a notice, that multiples of the identity are ignored, since unphysical

                else:
                    raise NotImplementedError(
                        "As of now we can only handle hamiltonians with at most two-qubit terms")

            if gamma_singles:
                gammas_singles.append(gamma_singles)
            if gamma_pairs:
                gammas_pairs.append(gamma_pairs)

            for qubit in reg:              # not efficient, but following the
                beta.append((1 - t / time) * dt)   # same logic as above
            betas.append(beta)

        # if there are no one qubit terms return a list containing an empty list
        # same for no two qubit terms. This ensures that prepare_qaoa_ansatz
        # works as expected
        if not gammas_singles:
            gammas_singles = [[]]
        if not gammas_pairs:
            gammas_pairs = [[]]

        # wrap it all nicely in a qaoa_parameters object
        params = GeneralQAOAParameters((reg, qubits_singles, qubits_pairs, timesteps),
                                       (betas, gammas_singles, gammas_pairs))
        return params

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.betas, label="betas", marker="s", ls="")
        if not _is_list_empty(self.gammas_singles):
            ax.plot(self.gammas_singles,
                    label="gammas_singles", marker="^", ls="")
        if not _is_list_empty(self.gammas_pairs):
            ax.plot(self.gammas_pairs, label="gammas_pairs", marker="v", ls="")
        ax.set_xlabel("timestep")
        ax.legend()


class AlternatingOperatorsQAOAParameters(GeneralQAOAParameters):
    """
    QAOA parameters that implement exp(-i*beta_f*H_0)*exp(-i*gamma_f*H_c)*...
    with an arbitrary H_c.

    Todo
    ----
    Typeset the eqation nicely?
    """

    def __repr__(self):
        string = "register: " + str(self.reg) + "\n"
        string += "betas: " + str(self._betas) + "\n"
        string += "qubits_singles: " + str(self.qubits_singles) + "\n"
        string += "gammas_singles: " + str(self._gammas_singles) + "\n"
        string += "qubits_pairs: " + str(self.qubits_pairs) + "\n"
        string += "gammas_pairs: " + str(self._gammas_pairs) + "\n"
        return(string)

    def __len__(self):
        return self.timesteps * 3

    def set_constant_parameters(self, constant_parameters: Tuple):
        """
        Parameters
        ----------
        constant_parameters: Tuple
            A tuple of the form ``(reg, qubits_singles, qubits_pairs, hamiltonian)``

        """
        self.reg, self.qubits_singles, self.qubits_pairs, self.timesteps, hamiltonian\
            = constant_parameters
        self.single_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 1]
        self.pair_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 2]

        if len(self.single_qubit_coeffs) != len(self.qubits_singles):
            raise ValueError("qubits_singles must have the same length as the"
                             "number of single qubit terms in the hamiltonian")
        if len(self.pair_qubit_coeffs) != len(self.qubits_pairs):
            raise ValueError("qubits_pairs must have the same length as the"
                             "number of two qubit terms in the hamiltonian")

    def update_variable_parameters(self, variable_parameters : Tuple =None):
        if variable_parameters is not None:
            self._betas, self._gammas_singles, self._gammas_pairs\
                = variable_parameters
            # check that the datas are good
            if self.timesteps != len(self._betas):
                raise ValueError(
                    "Please make all your angle arrays the same length!")
            if self.timesteps != len(self._gammas_singles):
                raise ValueError(
                    "Please make all your angle arrays the same length!")
            if self.timesteps != len(self._gammas_pairs):
                raise ValueError(
                    "Please make all your angle arrays the same length!")

        self.betas = [[b] * len(self.reg) for b in self._betas]
        self.gammas_singles = [[gamma * coeff for coeff in self.single_qubit_coeffs]
                               for gamma in self._gammas_singles]
        self.gammas_pairs = [[gamma * coeff for coeff in self.pair_qubit_coeffs]
                             for gamma in self._gammas_pairs]

    def update(self, new_values):
        # overwrite betas with new ones
        self._betas = list(new_values[0:self.timesteps])
        new_values = new_values[self.timesteps:]    # cut betas from new_values
        self._gammas_singles = list(new_values[0:self.timesteps])
        new_values = new_values[self.timesteps:]
        self._gammas_pairs = list(new_values[0:self.timesteps])
        new_values = new_values[self.timesteps:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new gammas and betas out of"
                                 "didn't have the right length!")
        self.update_variable_parameters()

    def raw(self):
        raw_data = []
        raw_data += self._betas
        raw_data += self._gammas_singles
        raw_data += self._gammas_pairs
        return raw_data

    @classmethod
    def from_hamiltonian(cls,
                         cost_hamiltonian: PauliSum,
                         timesteps: int,
                         time: float = None,
                         reg: List = None):
        """
        Returns
        -------
        :rtype:     AlternatingOperatorsQAOAParameters
        A `AlternatingOperatorsQAOAParameters` object holding all the
        parameters
        """
        if time is None:
            time = float(0.7 * timesteps)
        if reg is None:
            reg = cost_hamiltonian.get_qubits()
        # create evenly spaced timesteps at the centers of #timesteps intervals
        dt = time / timesteps
        times = np.linspace(time * (0.5 / timesteps), time
                         * (1 - 0.5 / timesteps), timesteps)

        # fill qubits_singles and qubits_pairs according to the terms in the hamiltonian
        qubits_singles = []
        qubits_pairs = []
        for term in cost_hamiltonian:
            if len(term) == 1:
                qubits_singles.append(term.get_qubits()[0])
            elif len(term) == 2:
                qubits_pairs.append(term.get_qubits())
            elif len(term) == 0:
                pass  # could give a notice, that multiples of the identity are ignored, since unphysical
            else:
                raise NotImplementedError("As of now we can only handle"
                                    "hamiltonians with at most two-qubit terms")

        # fill betas, gammas_singles and gammas_pairs
        betas = [dt * (1 - t / time) for t in times]
        gammas_singles = [dt * t / time for t in times]
        gammas_pairs = [dt * t / time for t in times]

        # wrap it all nicely in a qaoa_parameters object
        params = AlternatingOperatorsQAOAParameters(
            (reg, qubits_singles, qubits_pairs, timesteps, cost_hamiltonian),
            (betas, gammas_singles, gammas_pairs))
        return params

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self._betas, label="betas", marker="s", ls="")
        if not _is_list_empty(self._gammas_singles):
            ax.plot(self._gammas_singles,
                    label="gammas_singles", marker="^", ls="")
        if not _is_list_empty(self._gammas_pairs):
            ax.plot(self._gammas_pairs, label="gammas_pairs", marker="v", ls="")
        ax.set_xlabel("timestep")
        # ax.grid(linestyle='--')
        ax.legend()


class AdiabaticTimestepsQAOAParameters(AbstractQAOAParameters):
    """
    QAOA parameters that implement
    U = exp(-i*(T-t_f)H_0)exp(-i*t_f*H_c) ... exp(-i*(T-t_0)H_0)exp(-i*t_0*H_c)
    and only vary the t_i

    Todo
    ----
    Typeset the equation nicely?
    """

    def __repr__(self):
        string = "times: " + str(self._times)
        return string

    def __len__(self):   # needs fixing
        return self.timesteps

    def set_constant_parameters(self, constant_parameters: Tuple):
        """
        Parameters
        ----------
        constant_parameters : Tuple
            A tuple of the form ``(reg, qubits_singles, qubits_pairs,
            hamiltonian, self.time)``
        """
        self.reg, self.qubits_singles, self.qubits_pairs, self.timesteps, hamiltonian, self._T\
            = constant_parameters
        self.single_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 1]
        self.pair_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 2]
        if len(self.single_qubit_coeffs) != len(self.qubits_singles):
            raise ValueError("qubits_singles must have the same length as the "
                             "number of single qubit terms in the hamiltonian")
        if len(self.pair_qubit_coeffs) != len(self.qubits_pairs):
            raise ValueError("qubits_pairs must have the same length as the "
                             "number of two qubit terms in the hamiltonian")

    def update_variable_parameters(self, variable_parameters: Tuple =None):
        if variable_parameters is not None:
            # check that the datas are good
            if self.timesteps != len(variable_parameters):
                raise ValueError(
                    "variable_parameters has the wrong length")
            self._times = variable_parameters
        
        dt = self._T / self.timesteps
        self.betas = [[(1 - t / self._T) * (dt)] * len(self.reg)
                      for i, t in enumerate(self._times)]
        self.gammas_singles = [[t * dt * coeff / self._T for coeff in self.single_qubit_coeffs]
                               for i, t in enumerate(self._times)]
        self.gammas_pairs = [[t * dt * coeff / self._T for coeff in self.pair_qubit_coeffs]
                             for i, t in enumerate(self._times)]

    def update(self, new_values):
        if len(new_values) != self.timesteps:
            raise RuntimeWarning(
                "the new times should have length timesteps+1")
        self._times = new_values

        self.update_variable_parameters()

    def raw(self):
        """
        Returns
        -------
        Union[List[float], np.array]:
            A list or array of the times `t_i`
        """
        return self._times

    @classmethod
    def from_hamiltonian(cls,
                         cost_hamiltonian: PauliSum,
                         timesteps: int,
                         time: float = None,
                         reg: List = None):
        """
        Returns
        -------
        AlternatingOperatorsQAOAParameters :
            A `AlternatingOperatorsQAOAParameters` object holding all the
            parameters
        """
        if reg is None:
            reg = cost_hamiltonian.get_qubits()
        if time is None:
            time = 0.7 * timesteps

        times = list(np.linspace(time * (0.5 / timesteps),
                                 time * (1 - 0.5 / timesteps), timesteps))

        # fill qubits_singles and qubits_pairs according to the terms in the hamiltonian
        qubits_singles = []
        qubits_pairs = []
        for term in cost_hamiltonian:
            if len(term) == 1:
                # needs fixing...
                qubits_singles.append(term.get_qubits()[0])
            elif len(term) == 2:
                qubits_pairs.append(term.get_qubits())
            elif len(term) == 0:
                pass  # could give a notice, that multiples of the identity are ignored, since unphysical
            else:
                raise NotImplementedError("As of now we can only handle "
                                    "hamiltonians with at most two-qubit terms")

        # wrap it all nicely in a qaoa_parameters object
        params = AdiabaticTimestepsQAOAParameters(
            (reg, qubits_singles, qubits_pairs, timesteps, cost_hamiltonian, time),
            (times))
        return params

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self._times, label="times", marker="s", ls="")
        ax.set_xlabel("timestep number")
        ax.legend()


class FourierQAOAParameters(AbstractQAOAParameters):
    """
    The QAOA parameters as the sine/cosine transform of the original gammas
    and betas. See ()[] for a detailled description.

    Todo
    ----
    Actually cite the paper.
    """

    def __repr__(self):
        string = "register: " + str(self.reg) + "\n"
        string += "u_singles: " + str(self._u_singles) + "\n"
        string += "u_pairs: " + str(self._u_pairs) + "\n"
        string += "v: " + str(self._v) + "\n"
        return(string)

    def __len__(self):
        return 3 * self.q

    def set_constant_parameters(self, constant_parameters):
        """
        Parameters
        ----------
        param constant_parameters:  Tuple
            A tuple containing ``(reg, qubits_singles, qubits_pairs, timesteps, hamiltonian,  q)``
        """
        self.reg, self.qubits_singles, self.qubits_pairs, self.timesteps,\
            hamiltonian, self.q = constant_parameters

        self.single_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 1]
        self.pair_qubit_coeffs = [
            term.coefficient.real for term in hamiltonian if len(term) == 2]

    def update_variable_parameters(self, variable_parameters=None):
        """
        Parameters
        ----------
        param variable_parameters:  Tuple[List[float], List[float], List[float]]
        A tuple containing ``(v,  u_singles, u_pairs)``
        """
        def _dst(v, p):
            """Compute the discrete sine transform from frequency to timespace."""
            x = np.zeros(p)
            for i in range(p):
                for k in range(len(v)):
                    x[i] += v[k] * np.sin((k + 0.5) * (i + 0.5) * np.pi / p)
            return x

        def _dct(u, p):
            """Compute the discrete cosine transform from frequency to timespace."""
            x = np.zeros(p)
            for i in range(p):
                for k in range(len(u)):
                    x[i] += u[k] * np.cos((k + 0.5) * (i + 0.5) * np.pi / p)
            return x

        if variable_parameters is not None:
            self._v, self._u_singles, self._u_pairs = variable_parameters

            # check that the datas are good
            if self.q != len(self._v):
                raise ValueError(
                    "Please make all your fourier coeff arrays the same length!")
            if self.q != len(self._u_singles):
                raise ValueError(
                    "Please make all your fourier coeff arrays the same length!")
            if self.q != len(self._u_pairs):
                raise ValueError(
                    "Please make all your fourier coeff arrays the same length!")

        self._betas = _dct(self._v, self.timesteps)
        self._gammas_singles = _dst(self._u_singles, self.timesteps)
        self._gammas_pairs = _dst(self._u_pairs, self.timesteps)

        self.betas = [[b] * len(self.reg) for b in self._betas]
        self.gammas_singles = [[gamma * coeff for coeff in self.single_qubit_coeffs]
                               for gamma in self._gammas_singles]
        self.gammas_pairs = [[gamma * coeff for coeff in self.pair_qubit_coeffs]
                             for gamma in self._gammas_pairs]

    def update(self, new_values):
        self._v = list(new_values[0:self.q])   # overwrite betas with new ones
        new_values = new_values[self.q:]    # cut betas from new_values
        self._u_singles = list(new_values[0:self.q])
        new_values = new_values[self.q:]
        self._u_pairs = list(new_values[0:self.q])
        new_values = new_values[self.q:]

        if not len(new_values) == 0:
            raise RuntimeWarning("list to make new u's and v's out of\
            didn't have the right length!")
        self.update_variable_parameters()

    def raw(self):
        raw_data = []
        raw_data += self._v
        raw_data += self._u_singles
        raw_data += self._u_pairs
        return raw_data

    @classmethod
    def from_hamiltonian(cls,
                         cost_hamiltonian: PauliSum,
                         timesteps: int,
                         q: int = 4,
                         time: float = None,
                         reg: List = None):
        """
        Parameters
        ----------
        cost_hamiltonian : PauliSum
            The cost hamiltonian
        timesteps: int
            number of timesteps
        time: Number
            total time. Set to 0.7*timesteps if None is passed.
        fourier: q
            Number of Fourier coeffs. Defaults to 4
        reg : List
            The qubits to apply X-Rotations on

        Returns
        -------
        AlternatingOperatorsQAOAParameters:
            A `AlternatingOperatorsQAOAParameters` object holding all the
            parameters

        ToDo
        ----
        Make a more informed choice of the default value for q. Probably
        depending on nqubits
        """
        # fill qubits_singles and qubits_pairs according to the terms in the hamiltonian

        if reg is None:
            reg = cost_hamiltonian.get_qubits()

        qubits_singles = []
        qubits_pairs = []
        for term in cost_hamiltonian:
            if len(term) == 1:
                qubits_singles.append(term.get_qubits()[0])
            elif len(term) == 2:
                qubits_pairs.append(term.get_qubits())
            elif len(term) == 0:
                pass  # could give a notice, that multiples of the identity are ignored, since unphysical
            else:
                raise NotImplementedError("As of now we can only handle "
                                "hamiltonians with at most two-qubit terms")

        if time is None:
            time = 0.7 * timesteps

        # fill betas, gammas_singles and gammas_pairs
        v = [time / timesteps, *[0] * (q - 1)]
        u_singles = [time / timesteps, *[0] * (q - 1)]
        u_pairs = [time / timesteps, *[0] * (q - 1)]

        # wrap it all nicely in a qaoa_parameters object
        params = FourierQAOAParameters(
            (reg, qubits_singles, qubits_pairs, timesteps, cost_hamiltonian, q),
            (v, u_singles, u_pairs))
        return params

    def plot(self, ax=None):
        warnings.warn("Plotting the gammas and betas through DCT and DST. If you are "
              "interested in v, u_singles and u_pairs you can access them via "
              "params._v, params._u_singles, params._v_pairs")
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self._betas, label="betas", marker="s", ls="")
        if not _is_list_empty(self._gammas_singles):
            ax.plot(self._gammas_singles,
                    label="gammas_singles", marker="^", ls="")
        if not _is_list_empty(self._gammas_pairs):
            ax.plot(self._gammas_pairs, label="gammas_pairs", marker="v", ls="")
        ax.set_xlabel("timestep")
        # ax.grid(linestyle='--')
        ax.legend()
