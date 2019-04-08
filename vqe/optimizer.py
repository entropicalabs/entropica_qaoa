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

def _reduce_noisy_cost_function(fun):
    def reduced(*args, **kwargs):
        return fun(nshots=1000, *args, **kwargs)[0]
    return reduced

def scipy_optimizer(cost_function : Callable[[Iterable], Tuple[float, float]],
                    params0 : Iterable,
                    epsilon : float =1e-5,
                    nshots: int =1000,
                    method="COBYLA",
                    **mininize_kwargs):
    """A scipy.optimize.minimize wrapper for VQE
    """
    fun = _reduce_noisy_cost_function(cost_function)
    out = minimize(fun, params0, method=method, tol=epsilon, **mininize_kwargs)
    return out
