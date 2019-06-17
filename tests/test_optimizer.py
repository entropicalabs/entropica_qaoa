"""
Test the optimizer implementation
"""

import os, sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


import numpy as np
from scipy.optimize import minimize
from vqe.optimizer import scipy_optimizer, scalar_cost_function

# def test_scipy_optimizer():
#     def test_cost_function(x, nshots):
#         noise = 1/np.sqrt(nshots)*np.random.randn()
#         out = (np.cos(x[0])*np.sin(x[1]) + noise, 1/np.sqrt(nshots))
#         return out

#     x0 = [0, -1]
#     res = scipy_optimizer(test_cost_function, x0, nshots=10000)
#     print(res)
#     assert np.allclose(res['fun'], -1, rtol=1.1)
#     assert np.allclose(res['x'], [0, -np.pi/2], rtol=1.5, atol=0.5)

def test_scalar_cost_function():

    @scalar_cost_function(nshots=10000)
    def test_cost_function(x, nshots):
        noise = 1/np.sqrt(nshots)*np.random.randn()
        out = (np.cos(x[0])*np.sin(x[1]) + noise, 1/np.sqrt(nshots))
        return out

    x0 = [0, -1]
    res = minimize(test_cost_function, x0)
    print(res)
    assert np.allclose(res['fun'], -1, rtol=1.1)
    assert np.allclose(res['x'], [0, -np.pi/2], rtol=1.5, atol=0.5)
