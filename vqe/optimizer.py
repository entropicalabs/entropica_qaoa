"""
Different implementations of the optimization routines of VQE.
For now we just wrap ``scipy.optimize.minimize``, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.
"""

from scipy.optimize import minimize
from functools import partial
from typing import Callable, Tuple, Iterable, Union, List, Dict
import numpy as np

# TODO decide, whether we really want to support cost_functions that return
# floats and ones that return tuples (exp_val, std_dev)
def scalar_cost_function(fun: Callable[[np.array, int], Tuple[float, float]],
                         nshots: int) -> Callable[[np.array], float]:
    """Decorator to make our cost_functions work with scalar minimizers.

    Parameters
    ----------
    fun:
        A cost_function that takes a parameter array ``params`` and number of
        shots ``nshots`` and returns a Tuple ``(expectation, standard_dev)``
    nshots:
        ``nshots`` argument of ``fun``

    Returns
    -------
    Callable[[np.array], float]:
        A cost_function that takes only the parameter array and returns only
        the expectation_value
    """
    def reduced(*args, **kwargs):
        out = fun(*args, nshots=nshots, **kwargs)
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
    fun = scalar_cost_function(cost_function, nshots=nshots)
    out = minimize(fun, params0, method=method, tol=epsilon, options=mininize_kwargs)
    return out
