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

Conversion functions for the different QAOA Parametrizations. So far we only
only support going from less to more specialiced parametrizations. The type
tree looks as follows:

   Extended   <-------- (FourierExtended)
       ^                      ^
       |                      |
StandardWithBias <------ FourierWithBias
       ^                      ^
       |                      |
    Standard  <----------- Fourier
       ^
       |
    Annealing
"""

from copy import deepcopy
import numpy as np

from entropica_qaoa.qaoa.parameters import (AnnealingParams,
                                            StandardParams,
                                            ExtendedParams,
                                            StandardWithBiasParams,
                                            FourierParams,
                                            FourierWithBiasParams)
from entropica_qaoa.qaoa._trig_transforms import dct, dst


def annealing_to_standard(params: AnnealingParams) -> StandardParams:
    out = deepcopy(params)
    out.__class__ = StandardParams
    out.betas = params._annealing_time * (1 - params.schedule) / params.n_steps
    out.gammas = params._annealing_time * params.schedule / params.n_steps

    # and clean up after us
    del out.__schedule
    del out._annealing_time
    return out


def standard_to_standard_w_bias(
        params: StandardParams) -> StandardWithBiasParams:
    out = deepcopy(params)
    out.__class__ = StandardWithBiasParams
    out.gammas_singles = params.gammas
    out.gammas_pairs = params.gammas

    # and clean up after us
    del out.__gammas
    return out


def standard_w_bias_to_extended(
        params: StandardWithBiasParams) -> ExtendedParams:
    out = deepcopy(params)
    out.__class__ = ExtendedParams
    out.betas = np.outer(params.betas, np.ones(len(params.reg)))
    out.gammas_singles = np.outer(params.gammas_singles,
                                  np.ones(len(params.qubits_singles)))
    out.gammas_pairs = np.outer(params.gammas_pairs,
                                np.ones(len(params.qubits_pairs)))
    return out

# ############################################################################
# Continue with testing this!
# ############################################################################
def fourier_to_standard(params: FourierParams) -> StandardParams:
    out = deepcopy(params)
    out._class__ = StandardParams
    out.betas = dct(params.v, out.n_steps)
    out.gammas = dst(params.u, out.n_steps)

    # and clean up
    del out.__u
    del out.__v
    del out.q
    return out
