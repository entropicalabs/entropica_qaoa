import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
from itertools import product
from typing import Iterable, Union, List, Tuple, Any, Type

"""
TODO:
    
    - At the moment all methods are hard-coded to plot the cost function, but should be generalised to allow 
    visualistion of the standard deviation too. Also, they might be combined onto a single plot
    
    - Be wary of there being too many attributes.
    
    - Methods that operate on the output of get_local_minima
"""

class landscape:
    
    def __init__(self,
                 the_params: Tuple,
                 param_ranges: Tuple,
                 func_vals: Tuple):
        
        self.the_parameters = the_params
        self.parameter_ranges = param_ranges
        self.func_vals, self.dev_vals = func_vals
        self.shape = np.shape(func_vals[0])
        self.local_mins = []
        self.larger_neighbours = []
        
        # Have a nice str or repr method for the class attributes
        
    ########### PLOTTING METHODS ##################
       
    def plot_contour(self): # Static or class method?
        
        """
        Contour plot of the landscape
        Needs to be cleaned up and better presented
        Figure out why this is useful
        """
        
        assert (len(self.the_parameters) == 2), "Two parameters required for visualisation on 3D plot."
        
        fig, ax = plt.subplots()
        
        param1, param2 = np.meshgrid(self.parameter_ranges[0],self.parameter_ranges[1],indexing='ij')
        
        contour_ = ax.contour(param1,param2,self.func_vals)
        ax.clabel(contour_, inline=1, fontsize=10)
        
    def plot_3D(self): # Static or class method?
#        
        assert (len(self.the_parameters) == 2), "Two parameters required for visualisation on 3D plot."
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    
        param1, param2 = np.meshgrid(self.parameter_ranges[0],self.parameter_ranges[1],indexing='ij')
        
        # Plot the surface.
        surf = ax.plot_surface(param1,param2,self.func_vals,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        
        # TODO: extract the parameter names, perhaps put the corresponding symbols as axes labels.
        plt.xlabel(str(self.the_parameters[0]))
        plt.ylabel(str(self.the_parameters[1]))
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
    def plot_image(self):

        """
        Plots an image of the cost function values
        TODO: label the plot axes correctly
        """
        plt.imshow(self.func_vals)
        plt.colorbar()
        plt.show()
#        
#    def plot_slice():
#        
    ########### LANDSCAPE ANALYSIS METHODS ##################
        
    def get_local_minima(self): # Static or class method?
    
        """
        Searches through an array to find locations of minima
        Since there may be multiple minima along any direction, a log(N) binary search cannot be used.
        This method exhaustively walks through element by element, comparing to neighbours
        """
        dims = self.shape
        inds_ranges = dims - np.ones(len(dims),)
        n_els = np.prod(dims)    
        
        iterables = [[-1,0,1]]*len(dims)
        neighb_array = [np.array(t) for t in product(*iterables)]
        n_neighbs = len(neighb_array)
        
        neighs_above = np.zeros(dims)
        for i in range(n_els):
    
            inds = np.unravel_index(i, tuple(dims))
            val_here = self.func_vals[tuple(inds)]
            
            neighb_vals = np.zeros((n_neighbs,))
            for j in range(n_neighbs):
                
                neighb_inds = inds + neighb_array[j]
                
                above_range = np.nonzero(neighb_inds > inds_ranges)
                below_range = np.nonzero(neighb_inds < 0)
                out_of_range = np.hstack((above_range,below_range))
                
                if len(out_of_range[0]) > 0 :
                    continue
                
                neighb_vals[j] = self.func_vals[tuple(neighb_inds)]
            
            larger_neighbs = sum(neighb_vals > val_here) 
    
            if larger_neighbs == n_neighbs-1:
                self.local_mins.append(inds) # Definite minimum
            
            neighs_above[tuple(inds)] = larger_neighbs
    
        self.larger_neighbours = neighs_above

        
#class landscape_compare:

