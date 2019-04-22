"""
Different implementations of the optimization routines of VQE.
For now we just wrap `scipy.optimize.minimize`, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.

TODO: Is it worth the hassle, to write a abstract_optimizer class that
provides a template and one concrete implementation that just wraps
`scipy.optimize.minimize`?
"""

from scipy.optimize import minimize
from functools import partial
from typing import Callable, Tuple, Iterable

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

def scipy_optimizer(cost_function : Callable[[Iterable], Tuple[float, float]],
                    params0 : Iterable,
                    epsilon : float =1e-5,
                    nshots: int =1000,
                    method="COBYLA",
                    **mininize_kwargs):
    """A scipy.optimize.minimize wrapper for VQE
    """
    fun = _reduce_noisy_cost_function(cost_function, nshots=nshots)
    out = minimize(fun, params0, method=method, tol=epsilon, options=mininize_kwargs)
    return out
