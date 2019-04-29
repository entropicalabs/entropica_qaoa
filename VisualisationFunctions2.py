import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

"""
JL's comments
-------------
Codestyle
~~~~~~~~~
For the docstrings I adhered to the numpy docstring style
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
because sphinx (the tool I plan to use for automatic documentation) works
well with this docstring style.

EM: OK, will do this more or less at the end.

For function, variable and class names I (mostly) adhere to the PEP8 style.
(https://www.python.org/dev/peps/pep-0008/), just so that it looks like any
other good python code out there. (ThisIsAClass, this_is_a_function, this_is_variable).
Would you mind renaming your classes and functions accordingly for consistency
across the code base (I know pycharm can do project wide renames. Only in the
notebooks we will have to do it manually :/)

EM: Done.


QAOA parametrizations
~~~~~~~~~~~~~~~~~~~~~
You have only gammas and betas, where all my QAOA parameter classes have
different gammas for the one qubit terms / bias terms and the two qubit terms /
coupling terms. (I call them gammas_singles and gammas_pairs). Do you want me
to implement a QAOAParameter class, that has only gammas and betas like you use them?
(Wouldn't take too long, since I just have to inherit from AbstractQAOAParameters and
copy paste the code from AlternatingOperatorsQAOAParameters with minor modifications)

EM: I think we need to make a decision wrt how the user inputs parameters, and how they are 
returned from the optimiser. That will guide how the parameters for the sweeps
are passed in.
    
"""

def plot_qaoa_parameters(parameters):
    """
    Plot the optimal QAOA parameters found in optimization

    :param parameters:   a subtype of qaoa.abstract_qaoa_parameters
    :rtype:              matplotlib axes object? None?
    """
    # implementation suggestion:
    parameters.plot()


def vqe_optimization_stacktrace_plot(ax=None):
    """
    Plot the cost_function values of a VQE run (and by extension of a QAOA run),
    vs. the iteration step in the optimization process.
    
    :TODO (JL):  Decide, whether to create the log in the cost_function or in the
                 optimizer. Does scipy.optimize.minimize allow logging of function
                 calls? 
            
    :TODO (EM):  I like the idea of plotting information related to function
                 calls. I wonder if we can keep track of the number of function
                 evaluations at each step. Also, for the steps where the most function 
                 evaluations had to be made in order to proceed, what the values of the 
                 parameters at those locations? 
                 I can imagine that might be useful somehow.
            
    """

def plot_networkx_graph(G):
    
    """
    Takes in a networkx graph and plots it
    """
    
    weights = [*nx.get_edge_attributes(G,'weight').values()]
    pos = nx.shell_layout(G)
    
    nx.draw(G,pos,node_color='#A0CBE2',with_labels=True,edge_color=weights,
                 width=4, edge_cmap=plt.cm.Blues)
    plt.show()
    

def plot_hamiltonian_eigvals(n_qubits,cost_hamiltonian):

    """
    Plot the energy landscape of the cost Hamiltonian (i.e the eigenstates of the cost function)
    """
    
    fig, ax = plt.subplots()
    
    format_str = '{0:0' + str(n_qubits) + 'b}'
    labels = [r'$\left|' +
              format_str.format(i) + r'\right>$' for i in range(2**n_qubits)]
   
    matrix = cost_hamiltonian.matrix()
    eigvals = matrix.diag()

    ax.set_xticklabels(labels, minor=False)
    ax.set_xlabel("State")
    ax.set_ylabel("Energy")

    plt.plot(eigvals)
    plt.show()

def plot_parametric_cost_function(params,cost):
    
    """
    Takes the parameters that have been swept over, and the corresponding values of the cost function,
    and plots the landscape. 
    
    QUESTIONS: How to present the graphs & what choices to give users.
    
    (1) Separate figures for cost & variance, or same figure, or user specifies?
    (2) 3D plots, or contour plots, or user specifies?
    etc
    
    """
    
def plot_optimal_string_probability(params,optimal_string):
    
    """
    For the specified parameters, plots the probability of the known optimal string
    as a function of the QAOA iterations
    
    The intended uses are:
        
        (1) If you just start anywhere in parameter space, how quickly does the known
        solution actually begin to appear, and with what probability?
        (2) With the known optimal parameter set, same question. 
        
    Ultimately, want to find heuristic ideas for cutting short the necessary circuit depth,
    or reducing the search space by focusing only on the most promising regions of parameter
    space. [BUT THIS WOULD BE A METHOD TO IMPLEMENT FOR A FUTURE RELEASE, FOR NOW WE CAN JUST PLOT THE GRAPHS]
        
    """