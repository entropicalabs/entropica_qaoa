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

   Extended   <--------- FourierExtended
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
from typing import Type

from copy import deepcopy
import numpy as np
from scipy.fftpack import dct, dst

# import entropica_qaoa.qaoa.parameters as P
#                                     # import (AnnealingParams,
#                                     #         StandardParams,
#                                     #         ExtendedParams,
#                                     #         StandardWithBiasParams,
#                                     #         FourierParams,
#                                     #         FourierWithBiasParams,
#                                     #         FourierExtendedParams)
# (AnnealingParams, StandardParams, StandardWithBiasParams,
#  ExtendedParams, FourierParams, FourierWithBiasParams,
#  FourierExtendedParams)\
# = (P.AnnealingParams, P.StandardParams, P.StandardWithBiasParams,
#    P.ExtendedParams, P.FourierParams, P.FourierWithBiasParams,
#    P.FourierExtendedParams)

from entropica_qaoa.qaoa.parameters import (AnnealingParams,
                                            StandardParams,
                                            ExtendedParams,
                                            StandardWithBiasParams,
                                            FourierParams,
                                            FourierWithBiasParams,
                                            FourierExtendedParams)


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


def fourier_to_standard(params: FourierParams) -> StandardParams:
    out = deepcopy(params)
    out.__class__ = StandardParams
    out.betas = dct(params.v, n=out.n_steps)
    out.gammas = dst(params.u, n=out.n_steps)

    # and clean up
    del out.__u
    del out.__v
    del out.q

    return out


def fourier_w_bias_to_standard_w_bias(
        params: FourierWithBiasParams) -> StandardWithBiasParams:
    out = deepcopy(params)
    out.__class__ = StandardWithBiasParams
    out.betas = dct(params.v, n=out.n_steps)
    out.gammas_singles = dst(params.u_singles, n=out.n_steps)
    out.gammas_pairs = dst(params.u_pairs, n=out.n_steps)

    # and clean up
    del out.__u_singles
    del out.__u_pairs
    del out.__v
    del out.q

    return out


def fourier_to_fourier_w_bias(params: FourierParams) -> FourierWithBiasParams:
    out = deepcopy(params)
    out.__class__ = FourierWithBiasParams
    out.u_singles = params.u
    out.u_pairs = params.u

    # and clean up
    del out.__u

    return out


def fourier_w_bias_to_fourier_extended(
        params: FourierWithBiasParams) -> FourierExtendedParams:
    out = deepcopy(params)
    out.__class__ = FourierExtendedParams
    out.v = np.outer(params.v, np.ones(len(params.reg)))
    out.u_singles = np.outer(params.u_singles,
                             np.ones(len(params.qubits_singles)))
    out.u_pairs = np.outer(params.u_pairs,
                           np.ones(len(params.qubits_pairs)))
    return out


def fourier_extended_to_extended(
        params: FourierExtendedParams) -> ExtendedParams:
    out = deepcopy(params)
    out.__class__ = ExtendedParams
    out.betas = dct(params.v, n=out.n_steps, axis=0)
    out.gammas_singles = dst(params.u_singles, n=out.n_steps, axis=0)
    out.gammas_pairs = dst(params.u_pairs, n=out.n_steps, axis=0)

    # and clean up
    del out.__u_singles
    del out.__u_pairs
    del out.__v
    del out.q

    return out


# #############################################################################
# And now all the possible compositions as well:
# Todo: Create this code automatically by traversing the tree?
# #############################################################################

def annealing_to_standard_w_bias(
        params: AnnealingParams) -> StandardWithBiasParams:
    return standard_to_standard_w_bias(annealing_to_standard(params))


def annealing_to_extended(
        params: AnnealingParams) -> ExtendedParams:
    return standard_w_bias_to_extended(annealing_to_standard_w_bias(params))


def standard_to_extended(
        params: StandardParams) -> ExtendedParams:
    return standard_w_bias_to_extended(standard_to_standard_w_bias(params))


def fourier_to_fourier_extended(
        params: FourierParams) -> FourierExtendedParams:
    return fourier_w_bias_to_fourier_extended(
                fourier_to_fourier_w_bias(params))


def fourier_to_standard_w_bias(
        params: FourierParams) -> StandardWithBiasParams:
    return standard_to_standard_w_bias(fourier_to_standard(params))


def fourier_to_extended(
        params: FourierParams) -> ExtendedParams:
    return standard_w_bias_to_extended(fourier_to_standard_w_bias(params))


def fourier_w_bias_to_extended(
        params: FourierWithBiasParams) -> ExtendedParams:
    return standard_w_bias_to_extended(
            fourier_w_bias_to_standard_w_bias(params))


# A dict with all the conversion functions accessible by
# (input_type, output_type)
conversion_functions =\
    {
     (FourierExtendedParams, ExtendedParams): fourier_extended_to_extended,
     (FourierWithBiasParams, StandardWithBiasParams):\
     fourier_w_bias_to_standard_w_bias,
     (FourierParams, StandardParams): fourier_to_standard,
     (AnnealingParams, StandardParams): annealing_to_standard,
     (StandardParams, StandardWithBiasParams): standard_to_standard_w_bias,
     (StandardWithBiasParams, ExtendedParams): standard_w_bias_to_extended,
     (FourierParams, FourierWithBiasParams): fourier_to_fourier_w_bias,
     (FourierWithBiasParams, FourierExtendedParams):\
     fourier_w_bias_to_fourier_extended,
     (AnnealingParams, StandardWithBiasParams): annealing_to_standard_w_bias,
     (AnnealingParams, ExtendedParams): annealing_to_extended,
     (StandardParams, ExtendedParams): standard_to_extended,
     (FourierParams, FourierExtendedParams): fourier_to_fourier_extended,
     (FourierParams, StandardWithBiasParams): fourier_to_standard_w_bias,
     (FourierParams, ExtendedParams): fourier_to_extended,
     (FourierWithBiasParams, ExtendedParams): fourier_w_bias_to_extended,
    }


def converter(params, target_type: type):
    """
    Convert ``params`` to type ``target_type``

    Parameters
    ----------
    params:
        The input parameters
    target_type:
        The target type

    Returns
    -------
    target_type:
        The converted parameters

    Raises
    ------
    TypeError:
        If conversion from type(params) to target_type is not supported.
    """
    try:
        out = conversion_functions[(type(params), target_type)](params)
        return out
    except KeyError:
        raise TypeError(f"Conversion from {type(params)} to {target_type} "
                        "not supported.")
