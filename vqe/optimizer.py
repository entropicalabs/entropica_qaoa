"""
Different implementations of the optimization routines of VQE.
For now we just wrap ``scipy.optimize.minimize``, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.

Todo
----
Is it worth the hassle, to write a abstract_optimizer class that
provides a template and one concrete implementation that just wraps
`scipy.optimize.minimize`?
"""

from scipy.optimize import minimize
from functools import partial
from typing import Callable, Tuple, Iterable, Union
import numpy as np

# TODO decide, whether we really want to support cost_functions that return
# floats and ones that return tuples (exp_val, std_dev)
def _reduce_noisy_cost_function(fun, nshots):
    def reduced(*args, **kwargs):
        out = fun(*args, nshots=nshots, **kwargs)
        try:
            return out[0]
        except (TypeError, IndexError):
            return out
    return reduced

def scipy_optimizer(cost_function : Callable[Union[List[float], np.array], Tuple[float, float]],
                    params0 : Union[List[float], np.array],
                    epsilon : float =1e-5,
                    nshots: int =1000,
                    method="COBYLA",
                    **mininize_kwargs):
    """A ``scipy.optimize.minimize` wrapper for VQE.

    Parameters
    ----------
    param cost_function: Callable[Union[List[float], np.array], Tuple[float, float]]
        The cost function to minimize. It takes a list of floats or numpy array
        as parameters and returns a tuple ``(expectation_value, standard_deviation)``
    param params0: Union[List[float], np.array]
        The initial parameters for the optimization
    param epsilon: float
        The desired accuracy for the function value after optimization
    nshots: int
        The number of shots to take for each function evaluation of the cost_function
    param method : string
        The optimizer to use. Can be any of the optimizer included in
        ``scipy.optimize.minimize``. Default is `Cobyla`.
    param minimize_kwargs:
        The keyword arguments passed forward to ``scipy.optimize.minimize``.
        See its documentation for the options.

    Returns
    -------
    Dict :
        The output of ``scipy.optimize.minimize`` after minimization
    """
    fun = _reduce_noisy_cost_function(cost_function, nshots=nshots)
    out = minimize(fun, params0, method=method, tol=epsilon, options=mininize_kwargs)
    return out
