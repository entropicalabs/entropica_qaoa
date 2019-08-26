#   Copyright 2019 Entropica Labs
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
Implementation of the discrete sine and cosine transforms used in the Fourier
Parametrizations

Todo
----
Replace them with properly vectorized versions or versions from scipy.fftpack
"""

import numpy as np


def dst(v, p):
    """Compute the discrete sine transform from frequency to timespace."""
    x = np.zeros(p)
    for i in range(p):
        for k in range(len(v)):
            x[i] += v[k] * np.sin((k + 0.5) * (i + 0.5) * np.pi / p)
    return x


def dct(u, p):
    """Compute the discrete cosine transform from frequency to timespace."""
    x = np.zeros(p)
    for i in range(p):
        for k in range(len(u)):
            x[i] += u[k] * np.cos((k + 0.5) * (i + 0.5) * np.pi / p)
    return x
