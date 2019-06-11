#---------------------------IMPORTS----------------------------
import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from math import log

#imports needed to run QAOA
from typing import Type, Iterable

# import the neccesary pyquil modules
from qaoa.cost_function import QAOACostFunctionOnQVM, QAOACostFunctionOnWFSim
from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm

# import the QAOAParameters that we want to demo
from qaoa.parameters import AdiabaticTimestepsQAOAParameters,\
AlternatingOperatorsQAOAParameters, AbstractQAOAParameters, GeneralQAOAParameters,\
QAOAParameterIterator, FourierQAOAParameters

from vqe.optimizer import scipy_optimizer


#---------------------------FUNCTIONS------------------------
def generate_clusters(n_points, mean_center_dist, std_x, std_y, cov=0):
    '''
    Generate two clusters of data around two centers that are a predefined distance (mean_center_dist) apart.
    Each cluster has 'n_points' which are distributed normally around the center with standard deviation x and y as given.
    Returns a dataframe of the two clusters
    '''
    #define group labels and their centers
    groups = {0: (0,0),
              1: (mean_center_dist,0)}

    #create labeled x and y data
    data = pd.DataFrame(index=range(n_points*len(groups)), columns=['x','y','label'])
    for i, group in enumerate(groups.keys()):
        #randomly select n datapoints from a gaussian distrbution
        data.loc[i*n_points:((i+1)*n_points)-1,['x','y']]\
            = np.random.multivariate_normal(
                groups[group],
                [[std_x[i]**2,cov],[cov,std_y[i]**2]],
                n_points)
                #number of clusters is 2 by QUBO
        #add group labels
        data.loc[i*n_points:((i+1)*n_points)-1,['label']] = group

    return data

def plot_clusters(data):
    '''
    Plot two clusters of data
    '''
    #set font size of labels on matplotlib plots
    plt.rc('font', size=16)
    #set style of plots
    sns.set_style('white')
    #define a custom palette
    customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']
    sns.set_palette(customPalette)
    #sns.palplot(customPalette)
    #plot data with seaborn
    facet = sns.lmplot(data=data, x='x', y='y', hue='label',
                       fit_reg=False, legend=True, legend_out=True)

def generate_hamiltonian(dist):
    pauli_list = list()
    m,n = dist.shape

    #pairwise interactions
    for i in range(m):
        for j in range(n):
            if i < j:
                term = PauliTerm("Z",i,dist.values[i][j])*PauliTerm("Z",j, 1.0)
                pauli_list.append(term)

    #Due to bug you need to put in at least one bias term, we choose to set to zero
    pauli_list.append(PauliTerm("Z",0,0.0))

    return PauliSum(pauli_list)

def run_qaoa(hamiltonian,timesteps=2,end_time=1,max_iters=150,init_state=None):
    params = GeneralQAOAParameters.linear_ramp_from_hamiltonian(hamiltonian,timesteps,time=end_time)

    sim = WavefunctionSimulator()

    cost_function = QAOACostFunctionOnWFSim(hamiltonian,
                                            params=params,
                                            sim=sim,
                                            return_standard_deviation=True,
                                            noisy=False,
                                            log=[],
                                            initial_state=init_state)

    res = scipy_optimizer(cost_function, params.raw(), epsilon=1e-3,
                          maxiter=max_iters)

    return cost_function.get_wavefunction(params.raw()), res

def return_lowest_state(probs):
    index_max = max(range(len(probs)), key=probs.__getitem__)
    string = '{0:0'+str(int(log(len(probs),2)))+'b}'
    string = string.format(index_max)
    return [int(item) for item in string]

#--------------------------SCRIPT----------------------------

if __name__ == "main":
    #Hyperparameters for data generation
    n_points = 5 #this is the number of points per cluster
    # therefore the number of qubits required : num_q = n_points*2
    distance = 2
    std_xs = [1,1]
    std_ys = [1,1]

    #Hyperparameters for QAOA
    timesteps = 3
    iters = 3000

    #Generate clusters
    generated = generate_clusters(n_points, mean_center_dist=distance, std_x=std_xs, std_y=std_ys,cov=0)

    data = generated.drop(columns=['label']).copy()

    #Generate Euclidean distance matrix
    dist = pd.DataFrame(distance_matrix(data.values,data.values,p=2),
                           index=data.index,columns=data.index)

    hamiltonian = generate_hamiltonian(dist)

    #THIS IS WHERE WE RUN QAOA
    wave_func , res = run_qaoa(hamiltonian,timesteps=timesteps,max_iters=iters)

    lowest = return_lowest_state(wave_func.probabilities())

    #Printed analysis
    true_clusters = generated['label'].values.astype(int)
    print('True Labels of samples:',true_clusters)
    print('Lowest QAOA State:',lowest)

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(lowest,true_clusters)
    print('Accuracy of Original State:',acc*100,'%')

    #Complement bit string
    final_c = [0 if item == 1 else 1 for item in lowest]

    acc_c = accuracy_score(final_c,true_clusters)
    print('Accuracy of Complement State:',acc_c*100,'%')
