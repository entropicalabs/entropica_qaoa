"""
Different implementations of the optimization routines of VQE.
For now we just wrap `scipy.optimize.minimize`, but design the whole in such a
manner, that we can later swap it out with more sophisticated optimizers.

TODO: Is it worth the hassle, to write a abstract_optimizer class that
provides a template and one concrete implementation that just wraps
`scipy.optimize.minimize`?
"""

class optimizer(cost_function, params0, epsilon):
    """
    An optimizer class for VQE. Works with noisy cost_functions that take the
    number of shots as an argument and report the uncertainty of the function
    evaluation.
    """
    # cost_function(params, nshots) -> (cost, sigma_cost)
    # some optimizer of noisy functions which take the number of shots as
    # arguments. Optimizes until changes are smaller than epsilon
