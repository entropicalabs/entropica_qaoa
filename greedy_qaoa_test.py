#Import libraries 
import os, glob
import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score



#Imports to run forest-qaoa
from typing import Type, Iterable

# import the neccesary pyquil modules
from qaoa.cost_function import QAOACostFunctionOnQVM, QAOACostFunctionOnWFSim
from pyquil.api import local_qvm, WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm
from pyquil import Program
from pyquil.gates import I,X

# import the QAOAParameters that we want to demo
from qaoa.parameters import AdiabaticTimestepsQAOAParameters,\
AlternatingOperatorsQAOAParameters, AbstractQAOAParameters, GeneralQAOAParameters,\
QAOAParameterIterator, FourierQAOAParameters

from vqe.optimizer import scipy_optimizer

#----------------------------------------
def pca_results(data, pca):    
    # Dimension indexing
    dimensions = ['PC-{}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys()) 
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights") 
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios# 
    for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

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
                                            init_state=init_state)
    
    res = scipy_optimizer(cost_function, params.raw(), epsilon=1e-3,
                          maxiter=max_iters)
    
    return cost_function.get_wavefunction(params.raw()), res

def return_lowest_state(probs):
    index_max = max(range(len(probs)), key=probs.__getitem__)
    string = '{0:0'+str(int(log(len(probs),2)))+'b}'
    return string.format(index_max)

#This function will return the distance matrix of the data, as projected onto the axis of each principal component
def return_pca_dists(data,pca):
    pc_dict = {}
    for i in range(pca.n_components):
        projections = list()
        for idx, row in data.iterrows():
            projections.append(np.dot(row,pcs[i]))
        
        projections = np.copy(projections).reshape(len(data),1)
        pc_dict['PC-'+ str(i+1)] = pd.DataFrame(distance_matrix(projections,projections,p=1),
                    index=data.index,columns=data.index)
    return pc_dict

#Import data


pickle_dir = '../../data/data.pkl'
df = pd.read_pickle(pickle_dir)

pca = PCA(n_components=5).fit(data)
pc_eigens = np.copy(pca.explained_variance_)
pcs = pca.components_

num_q = 10
timesteps = 5
iters = 500

num_qubits.append(num_q)
num_timesteps.append(timesteps)
num_iters.append(iters)

q_df = data.sample(num_q) #take any 'num_qubits' points from our df
dist = pd.DataFrame(distance_matrix(q_df.values,q_df.values,p=2),
                       index=q_df.index,columns=q_df.index)
hamiltonian = generate_hamiltonian(dist)

wave_func , res = run_qaoa(hamiltonian,timesteps=timesteps,max_iters=iters)
lowest = return_lowest_state(wave_func.probabilities())
lowest = [int(item) for item in lowest]
#The index values of the qubit - we can use to reconstruct
q_df.index.values
#The true clusters are the true values of each qubit
true_clusters = [df['target'].iloc[val] for val in q_df.index.values]

acc = accuracy_score(lowest,true_clusters)
#Complement bit string
final_c = [0 if item == 1 else 1 for item in lowest]
acc_c = accuracy_score(final_c,true_clusters)

qaoa_string.append(return_lowest_state(wave_func.probabilities()))
qaoa_acc.append(acc)
index.append(q_df.index.values)
true_labels.append(''.join([str(item) for item in true_clusters]))

pc_dict = return_pca_dists(q_df,pca)

#Create dictionary of hamiltonians based on the  distance matricies 
ham_dict = {}
for key in pc_dict:
    ham_dict[key] = generate_hamiltonian(pc_dict[key])
    
naive_wave_dict ={}
naive_res_dict = {}
for key in ham_dict:
    wave_func , res = run_qaoa(ham_dict[key],timesteps=timesteps,max_iters=iters)
    naive_wave_dict[key] = np.copy(wave_func.probabilities())
    naive_res_dict[key] = res.copy()
    
binaries = list()
for key in naive_wave_dict:
    binaries.append(return_lowest_state(naive_wave_dict[key]))
    
#The coefficients which will multiply each lowest energy state is the 'weighted' of each eigenvalue
eigens = pc_eigens / sum(pc_eigens)

    
#Turn into numpy array
binaries = np.copy(binaries)
#Turn the binaries into a into list of component-wise
bins = list()

#Multiply each item in the list by their respective eigenvalue coefficient
for i, binary in enumerate(binaries):
    bins.append([eigens[i]*int(item) for item in list(binary)])
bins = np.copy(bins)
for row in bins:
    print(row)
final = list()
for row in bins.T:
    final.append(int(round(row.sum())))
    
#The index values of the qubit - we can use to reconstruct
#The true clusters are the true values of each qubit
true_clusters = [df['target'].iloc[val] for val in q_df.index.values]
acc = accuracy_score(final,true_clusters)
#Complement bit string
final_c = [0 if item == 1 else 1 for item in final]
acc_c = accuracy_score(final_c,true_clusters)

naive_string.append(''.join(map(str, final)))
naive_acc.append(acc)

greedy_wave_dict ={}
greedy_res_dict = {}
init_state = None
for key in ham_dict:
    wave_func , res = run_qaoa(ham_dict[key],timesteps=5,
                               max_iters=500,init_state=init_state)
    greedy_wave_dict[key] = np.copy(wave_func.probabilities())
    greedy_res_dict[key] = res.copy()
    init_state = return_lowest_state(greedy_wave_dict[key])
final_state = [int(item) for item in init_state]

#The true clusters are the true values of each qubit
true_clusters = [df['target'].iloc[val] for val in q_df.index.values]
acc = accuracy_score(final_state,true_clusters)

#Complement bit string
final_c = [0 if item == 1 else 1 for item in final_state]

acc_c = accuracy_score(final_c,true_clusters)
greedy_string.append(''.join(map(str, final_state)))
greedy_acc.append(acc)

greedy_wave_dict ={}
greedy_res_dict = {}
init_state = None
for key in reversed(list(ham_dict.keys())):
    wave_func , res = run_qaoa(ham_dict[key],timesteps=5,
                               max_iters=500,init_state=init_state)
    greedy_wave_dict[key] = np.copy(wave_func.probabilities())
    greedy_res_dict[key] = res.copy()
    init_state = return_lowest_state(greedy_wave_dict[key])
final_state = [int(item) for item in init_state]
#The index values of the qubit - we can use to reconstruct
q_df.index.values
#The true clusters are the true values of each qubit
true_clusters = [df['target'].iloc[val] for val in q_df.index.values]

acc = accuracy_score(final_state,true_clusters)


#Complement bit string
final_c = [0 if item == 1 else 1 for item in final_state]

acc_c = accuracy_score(final_c,true_clusters)
greedy_string_pc1_last.append(''.join(map(str, final_state)))
greedy_acc_pc1_last.append(acc)

exports = {'naive_greedy_string': naive_string,
           'greedy_string': greedy_string,
           'greedy_string_pc1_last': greedy_string_pc1_last,
           'qaoa_string': qaoa_string,
           'naive_greedy_acc':naive_acc,
           'greedy_acc': greedy_acc,
           'greedy_acc_pc1_last': greedy_acc_pc1_last,
           'qaoa_acc':qaoa_acc,
           'true_labels': true_labels,
           'index':index,
           'num_qubits':num_qubits,
           'num_timesteps':num_timesteps,
           'num_iters':num_iters,
        }
#Create the global dataframe    
old = pd.read_pickle('../../data/qaoa_tests.pkl')
df = pd.DataFrame(exports)
old = old.append(df)
old.to_pickle('../../data/qaoa_tests.pkl')