import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from pyquil import Program
from pyquil.gates import RX, PHASE, CPHASE
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm


"""
MAJOR TODOs:
    
    1) Make things object oriented, integrating into the class structure of the whole package
    2) Integrate all the highlighted functions with JL's versions
    3) Abstract PlotParametricCostFunction, PlotParametricVariance etc into one method. The user can then specify which of a set of pre-defined common functions (cost, variance, etc)
      they want, or define their own operator whose expectation value they want to compute at each point in the parameter landscape.
    4) Users may want to work with several different landscape projections at once. Thus we need different instances, each characterised by the parameters being varied with the others
    fixed. This allows them to then call other abstract functions on each landscape projection - eg the PlotOptimalTrajectory method, etc.

"""


def BuildQAOACircuit(n_qubits,QAOA_p,qubits,coefficients,beta_angles,gamma_angles):
    
    """
    JL will already have a version of this
    """
    
    p = Program()

    for i in range(QAOA_p):
        
        betas = beta_angles[i*n_qubits:(i+1)*n_qubits]
        for j in range(n_qubits):
            
            p += RX(betas[j],j)
    
        gammas = gamma_angles[i*len(coefficients):(i+1)*len(coefficients)]
    
        for k in range(len(qubits)):
    
            term_coeff = gammas[k]*coefficients[k]
            qbits = qubits[k]
    
            if len(qbits) == 1:
    
                qb = qbits[0]
                p += PHASE(term_coeff,qb)
    
            elif len(qbits) == 2:
    
                qb1 = qbits[0]
                qb2 = qbits[1] 
    
                p += PHASE(term_coeff,qb1)
                p += PHASE(term_coeff,qb2)
                p += CPHASE(-2*term_coeff,qb1,qb2)   

    return p

def ApplyQAOACircuit(circuit):
    
    """
    JL will already have a version of this
    """
    
    wf_sim = WavefunctionSimulator()
    wavefn = wf_sim.wavefunction(circuit)
    
    return wavefn

def BuildCostFn_Pyquil(qubits,coefficients):

    """
    JL will already have a version of this
    """

    CostPaulis = []
    for i in range(len(qubits)):
        
        qbits = qubits[i]
        
        if len(qbits) == 1:
                
            qbit = qbits[0] 
    
            term = PauliTerm("Z",qbit,coefficients[i])
    
            CostPaulis.append(term)
            
        elif len(qbits) == 2:
    
            qb1 = qbits[0]
            qb2 = qbits[1] 
                
            term1 = PauliTerm("Z",qb1,coefficients[i])
            term2 = PauliTerm("Z",qb2,1.0)
            
            CostPaulis.append(term1*term2)
             
    CostPauliSum = PauliSum(CostPaulis)
            
    return CostPauliSum

def EvaluateCostFunction(circuit,hamiltonian):
    
    """
    JL will already have a version of this
    """

    wf_sim = WavefunctionSimulator()
    Exp_Val = wf_sim.expectation(circuit,hamiltonian)
    
    return Exp_Val

def EvaluateCostFunctionVariance(circuit,hamiltonian):
    
    """
    Computes the variance of the cost function wrt the state prepared by circuit.
    """

    wf_sim = WavefunctionSimulator()
    Variance = wf_sim.expectation(circuit,hamiltonian*hamiltonian) - (wf_sim.expectation(circuit,hamiltonian))**2
    
    return Variance
    
def CostFunctionMatrix(n_qubits, Operators_Dict,coefficients):
    
    """
    Builds the matrix representation of the cost function (which is just diagonal for our case, and is thus a vector). 
    The entries are thus the eigenvaules of the cost function.
    """
    Hamiltonian = -1*SumPauliOps(Operators_Dict,coefficients, n_qubits)
    
    return Hamiltonian
    
def PlotBareCostFunction(n_qubits, Operators_Dict, coefficients):
    
    """
    Plot the energy landscape of the cost Hamiltonian (i.e the eigenstates of the cost function)
    """
    
    Hamiltonian = CostFunctionMatrix(n_qubits, Operators_Dict,coefficients)
    
    x_var = [i for i in range(2**n_qubits)]
    plt.plot(x_var,Hamiltonian)
    plt.show()
    
def LandscapeParameters(n_qubits,qubits,coefficients,QAOA_p,betas2var,beta_range2var,betas,gammas2var,gamma_range2var,gammas):
    
    """
    Parameters
    ----------
    QAOA_p: integer
        Total number of QAOA steps
        
    betas2var: integer list
        The indices of beta parameters to vary, and in which QAOA step. List should be of the form [[beta_indices],[steps]]. 
        Eg to vary Beta_0 and Beta_4 (the mixer coefficients for qubits 0 and 4) in the first and 3rd steps respectively, betas2var = [[0,4],[1,3]]
        
    beta_range2var: list of arrays
        Each array specifies the range of values over which the corresponding betas term is to be varied. 
        E.g. beta_range2var = [np.linspace(0,np.pi),np.linspace(0,np.pi/2)]
    
    betas: list
        A list of the fixed values of all unvaried beta parameters at all QAOA steps, and arbitrary values for the parameters to be varied at the appropriate list indices.
    
    gammas_2var, gammas_range2var, gammas: as for the analogous beta parameters.
      
    Returns
    -------
    
    """
    
    beta_var, beta_p = betas2var
    gamma_var, gamma_p = gammas2var
    
    params2var = len(beta_var) + len(gamma_var)
    
    assert (params2var >= 1 and params2var <= 2), "Specified number of QAOA parameters to vary is zero or greater than two."
    assert all(i > 0 for i in beta_p + gamma_p), "QAOA step indices must be greater than zero"
    assert (max(list(beta_p) + list(gamma_p)) <= QAOA_p), "QAOA parameter specified to be varied in a step greater than the maximum number of steps."
        
    n_betas = int(len(betas)/QAOA_p)
    n_gammas = int(len(gammas)/QAOA_p)
    
    if beta_var:
        beta_p = [beta_p[i] - 1 for i in range(len(beta_p))]
        beta_var = [beta_var[i] + beta_p[i]*n_betas for i in range(len(beta_var))]      
    else:
        beta_range2var = []

    if gamma_var:
        gamma_p = [gamma_p[i] - 1 for i in range(len(gamma_p))]
        gamma_var = [QAOA_p*n_betas + gamma_var[i] + gamma_p[i]*n_gammas for i in range(len(gamma_var))]   
    else:
        gamma_range2var = []

    params = np.hstack((betas,gammas))  
    param1,param2 = beta_var + gamma_var  
    param1_range, param2_range = list(beta_range2var) + (gamma_range2var)
    
    return params, param1, param2, param1_range, param2_range
    
#def PlotParametricCostFunction(n_qubits,qubits,coefficients,hamiltonian,QAOA_p,betas2var,beta_range2var,betas,gammas2var,gamma_range2var,gammas):
def PlotParametricCostFunction(params, param1, param2, param1_range, param2_range,QAOA_p,qubits,coefficients,n_betas,hamiltonian,n_qubits):   
    
    """
    Plot the energy landscape of the cost Hamiltonian with respect to the parametric circuit parameters.
    
    TO DO
    -----
    - Include the case where only one parameter is specified, and plots a regular graph instead of a 3D figure.
       
    """
    n_betas = n_qubits
    
    cost = np.zeros((len(param1_range),len(param2_range)))
    for i in range(len(param1_range)):
        
        params[param1] = param1_range[i]
        
        for j in range(len(param2_range)):

            params[param2] = param2_range[j]
            
            betas = params[:QAOA_p*n_betas]
            gammas = params[QAOA_p*n_betas:]

            circuit = BuildQAOACircuit(n_qubits,QAOA_p,qubits,coefficients,betas,gammas) # This line is expensive and can be improved - probably eg using parametric functions etc
            cost[i,j] = EvaluateCostFunction(circuit,hamiltonian)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    param1, param2 = np.meshgrid(param1_range,param2_range,indexing='ij')
    
    # Plot the surface.
    surf = ax.plot_surface(param1,param2,cost,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # TODO: automatically generate the names of the parameters, eg Beta4
    plt.xlabel('Param1')
    plt.ylabel('Param2')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return cost

def PlotParametricVariance(params, param1, param2, param1_range, param2_range,QAOA_p,qubits,coefficients,n_betas,hamiltonian,n_qubits): 
    
    """
    Plot the variance of the energy landscape of the cost Hamiltonian with respect to the parametric circuit parameters
    
    TODO: investigate the warning message that appears in running this, e.g. 
    
        UserWarning: The term Z2Z0 will be combined with Z0Z2, but they have different orders of operations. 
        This doesn't matter for QVM or wavefunction simulation but may be important when running on an actual device.
    
        Can we suppress it if it really doesn't matter (which it shouldn't for this diagonal cost function)?
    
    """
    
    n_betas = n_qubits
    
    variance = np.zeros((len(param1_range),len(param2_range)))
    for i in range(len(param1_range)):
        
        params[param1] = param1_range[i]
        
        for j in range(len(param2_range)):

            params[param2] = param2_range[j]
            
            betas = params[:QAOA_p*n_betas]
            gammas = params[QAOA_p*n_betas:]

            circuit = BuildQAOACircuit(n_qubits,QAOA_p,qubits,coefficients,betas,gammas) # This line is expensive and can be improved - probably eg using parametric functions etc
            variance[i,j] = EvaluateCostFunctionVariance(circuit,hamiltonian)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    param1, param2 = np.meshgrid(param1_range,param2_range,indexing='ij')
    
    # Plot the surface.
    surf = ax.plot_surface(param1,param2,variance,cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    # TODO: automatically generate the names of the parameters, eg Beta4
    plt.xlabel('Param1')
    plt.ylabel('Param2')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    return variance
    
#def Plot_QAOA_Trajectory(LandscapeFunction,betas,gammas):
    
    """
    Plot how the QAOA path specified by the gammas and betas moves across the landscape projection onto a specified pair of parameters.
    
    Input:
        The pre-computed value of the cost function (or its variance, or some other function) in the landscape of interest [or it can be computed here for the first time?]
        The QAOA parameters for a given trajectory through the landscape
        
    Returns:
        Plots a graph showing how the trajetory moves through the landscape of the specified parameters
        
    """
    
    # assert(Specified parameters of interest are those to which the landscape being specified corresponds)
    
    

"""
OTHER FUNCTIONS THAT MIGHT BE NEEDED
"""

def BuildPauliOps(n_qubits,qubits):

    """
    Builds a dictionary of Pauli Z operators acting on one or two qubits (and identity on the rest).
    
    Input: 
        n_qubits: the total number of qubits in the system   
        qubits: a list of lists, with each sublist containing the qubit indices where the Z operator(s) for each term in the Hamiltonian act. 
                The left-most qubit has label 0, and the right-most has label n_qubits-1  [TODO: correct this to coincide with Forest's qubit enumeration, which is the reverse!]
    Returns:
        A dictionary of vectors, each corresponding to the diagonal of a corresponding Pauli operator (since all such operators are diagonal in the Z basis)      

    NB: The reason I think this is needed is because Pyquil does not have a way of turning a PauliSum or PauliTerm into a numpy array, so that its eigenvalues can be found.
    """
    
    Z = np.array([1,-1])

    Operators_Dict = {}
    for i in range(len(qubits)):

        qbits = qubits[i]
        
        if len(qbits) == 1:
            
            qbit = qbits[0] + 1

            left_I = np.ones(max(1,2**(qbit-1)))
            right_I = np.ones(max(1,2**(n_qubits - qbit)))
            
            final_op = np.kron(left_I,np.kron(Z,right_I))

            tmp_dict = {str(i): final_op}
            Operators_Dict.update(tmp_dict)
            
        elif len(qbits) == 2:

            qb1 = qbits[0] + 1
            qb2 = qbits[1] + 1
            
            if qb1 < qb2:
                left_I = np.ones(max(1,2**(qb1-1)))
                middle_I = np.ones(max(1,2**(qb2-1-qb1)))
                right_I = np.ones(max(1,2**(n_qubits - qb2)))
                
                right_op = np.kron(Z,right_I)
                left_op = np.kron(left_I,np.kron(Z,middle_I))
    
                operator = np.kron(left_op,right_op)
                
            elif qb1 > qb2:
                left_I = np.ones(max(1,2**(qb2-1)))
                middle_I = np.ones(max(1,2**(qb1-1-qb2)))
                right_I = np.ones(max(1,2**(n_qubits - qb1)))
                
                right_op = np.kron(Z,right_I)
                left_op = np.kron(left_I,np.kron(Z,middle_I))
            
                operator = np.kron(left_op,right_op)

            tmp_dict = {str(i): operator}
            Operators_Dict.update(tmp_dict)        
        
    return Operators_Dict

def SumPauliOps(Operators_Dict,coefficients,n_qubits):
    
    """
    Sums up the operators contained in Operators_Dict, mutliplying each by the corresponding entry in coefficients.
    """
    
    assert len(Operators_Dict) == len(coefficients)
    
    Pauli_Sum = np.zeros((2**(n_qubits),))
    for i in range(len(coefficients)):

        operator = Operators_Dict[str(i)]
        Pauli_Sum += coefficients[i]*operator
        
    return Pauli_Sum