"""
Different implementations of the optimization routines of VQE.
For now we just wrap ``scipy.optimize.minimize``, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.
"""

import warnings
from scipy.optimize import minimize
from functools import partial
from typing import Callable, Tuple, Iterable, Union, List, Dict
import numpy as np

# TODO decide, whether we really want to support cost_functions that return
# floats and ones that return tuples (exp_val, std_dev)
class scalar_cost_function():
    """Decorator to make our cost_functions work with scalar minimizers.

    Parameters
    ----------
    nshots:
        ``nshots`` argument of ``fun``

    Example
    -------
    Here goes the example code...
    """
    def __init__(self, nshots: int = 1000):
        self.nshots = nshots

    def _scalar_fun_decorator(self, fun):
        def reduced(*args, **kwargs):
            out = fun(*args, nshots=self.nshots, **kwargs)
            try:
                return out[0]
            except (TypeError, IndexError):
                return out
        return reduced

    def _scalar_class_decorator(self, obj):
        def reduced(*args, **kwargs):
            out = obj.__call__(*args, nshots=self.nshots, **kwargs)
            try:
                return out[0]
            except (TypeError, IndexError):
                return out
        obj.__call__ = reduced
        return obj

    def __call__(self,
                 fun: Callable[[np.array, int], Tuple[float, float]])\
            -> Callable[[np.array], float]:
        """Create the reduced scalar cost function.

        Parameters
        ----------
        fun : Callable[[np.array, int, Any], Tuple[float, float]]
            The original cost function with an ``nshots`` argument and
            ``(exp_val, std_dev)`` output.

        Returns
        -------
        Callable[[np.array], float]
            The reduced cost function without ``nshots`` argument and only
            ``exp_val`` output.
        """
        def reduced(*args, **kwargs):
            out = fun(*args, nshots=self.nshots, **kwargs)
            try:
                return out[0]
            except (TypeError, IndexError):
                return out
        return reduced


def scipy_optimizer(cost_function: Callable[[Union[List[float], np.array]], Tuple[float, float]],
                    params0: Union[List[float], np.array],
                    epsilon: float = 1e-5,
                    nshots: int = 1000,
                    method: str = "COBYLA",
                    **mininize_kwargs) -> Dict:
    """A ``scipy.optimize.minimize`` wrapper for VQE.

    Parameters
    ----------
    cost_function:
        The cost function to minimize. It takes a list of floats or numpy array
        as parameters and returns a tuple ``(expectation_value, standard_deviation)``
    params0:
        The initial parameters for the optimization
    epsilon:
        The desired accuracy for the function value after optimization
    nshots:
        The number of shots to take for each function evaluation of the cost_function
    method:
        The optimizer to use. Can be any of the optimizer included in
        ``scipy.optimize.minimize``. Default is `Cobyla`.
    minimize_kwargs:
        The keyword arguments passed forward to ``scipy.optimize.minimize``.
        See its documentation for the options.

    Returns
    -------
    Dict :
        The output of ``scipy.optimize.minimize`` after minimization
    """
    warnings.warn("scipy_optimizer is deprecated in favour of the "
                  "scalar_cost_function decorator. See its documentation for "
                  "details.",
                  DeprecationWarning)
    # strip the cost_function of the nshots argument and standard_deviation
    # @scalar_cost_function(nshots=nshots)
    # cost_function

    fun = scalar_cost_function(nshots=nshots)(cost_function)

    # and run the minimizer.
    out = minimize(fun, params0,
                   method=method, tol=epsilon, options=mininize_kwargs)
    return out
