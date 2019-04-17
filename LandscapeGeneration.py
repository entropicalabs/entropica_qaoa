import numpy as np
from numpy.random import rand, randint, choice


def DiscreteLandscape_Gaussian(n_qubits,mean,variance):
    
    """
    Generates a Hamiltonian where the mean bitstring energy is given by 'mean', 
    and the variance by 'variance'.
    """
    
def MultimodalParameterLandscape(args):
    
    """
    Generates a multimodal landscape in a 2D space of the QAOA parameters.
    Allows user to explore a hard problem instance where there are several deep
    and disconnected minima.
    
    Challenge: need to generate the landscape as a function of QAOA parameters,
    not just the eigenvalues of the Hamiltonian.
    
    One way to do this for two modes:
        
        Ensure that the energy of |10...> and |01...> are identical.
        Thus in terms of QAOA parameters E(beta_1 = pi, beta_2 = 0,...) is equal
        to E(beta_1 = 0, beta_2 = pi,...). These are two well-separated minima
        in the beta_1, beta_2 space. Now we can add a bias term to one of these
        qubits to make its valley slightly deeper. 
        
        
    """


