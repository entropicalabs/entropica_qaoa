# Copyright 2019 Entropica Labs
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
Different implementations of the optimization routines of VQE.
For now we just wrap ``scipy.optimize.minimize``, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.
"""

from scipy.optimize import minimize
from typing import Callable, Tuple, Union, List, Dict
import numpy as np
import warnings

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


def scipy_optimizer(cost_function: Callable[[Union[List[float], np.array]],
                                            Tuple[float, float]],
                    params0: Union[List[float], np.array],
                    epsilon: float = 1e-5,
                    nshots: int = 1000,
                    method: str = "COBYLA",
                    **mininize_kwargs) -> Dict:
    """A ``scipy.optimize.minimize`` wrapper for VQE.

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

    Note
    ----
    This method is deprecated in favor of using your optimizer of choice
    directly. To do so you have to make sure, that the cost function only
    returns a scalar and only takes in an array of parameters, by using the
    `scalar_cost_function` toggle in the cost function constructor and then replace the optimizer call: Replace

    >>> scipy_optimizer(cost_fn, params, epsilon=epsilon)

    with

    >>> scipy.optimize.minimize(cost_fn, params, tol=epsilon, method="Cobyla")

    """
    warnings.warn("This method is deprecated in favor of using "
                  "scipy.optimize.minimize directly. See the docstring "
                  "of this method, to see how to do this.", DeprecationWarning)
    fun = _reduce_noisy_cost_function(cost_function, nshots=nshots)
    out = minimize(fun, params0, method=method, tol=epsilon, options=mininize_kwargs)
    return out
