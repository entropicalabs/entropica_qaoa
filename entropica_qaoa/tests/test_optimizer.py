"""
Test the optimizer implementation
"""

import os, sys
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


import numpy as np
from entropica_qaoa.vqe.optimizer import scipy_optimizer

# TODO write a test involving the whole VQE stuff
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
