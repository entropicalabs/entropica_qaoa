import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

from typing import Iterable, Union, List, Tuple, Any, Type


class landscape:
    
    def __init__(self,
                 the_params: Tuple,
                 param_ranges: Tuple,
                 func_vals: Tuple):
        
        self.the_parameters = the_params
        self.parameter_ranges = param_ranges
        self.func_vals, self.dev_vals = func_vals
        self.shape = np.shape(func_vals)
        self.local_mins = []
        
        
        # Have a nice str or repr method for the class attributes

    def local_minima(self): # Static or class method?
        
        """
        Finds the list of local minima in the desired array
        signal.argrelmax finds maxima, so we find the maxima of -1*array
        signal.argrelmax returns any kind of minima (eg along any of the directions), not necessarily a minima in all directions
        we thus further analyse the result to obtain those points that are true local minima in all directions
        note that this method ignores points at the extremes of a given dimension
        
        TODO: systematically ensure this method is correct by constructing landscapes with known local minima & make sure it returns right result
        """
         
        mins = signal.argrelmax(-1*self.func_vals)
        mins = [[mins[0][i],mins[1][i]] for i in range(len(mins[0]))]
        
        local_mins = []
        for i in mins: 
            
            val_here = self.func_vals[tuple(i)] # Function value at coordinate i
            
            # Get indices of neighbouring points, checking if the point on the boundary of the coordinate system (ignore if so)
            edge_pts = [j for j in range(len(i)) if i[j] == self.shape[j]-1 or i[j] == 0]
            if edge_pts: 
                continue
            else:
                nbours = np.array([i]*4) + np.array([[-1,0], [0,-1], [1, 0], [0,1]]) # coordinates of neighbouring points (NEED TO EXTEND TO ARBITRARY DIMS)
            
            val_nbours = [self.func_vals[tuple(j)] for j in nbours] # Function values at neighbouring points

            if all(val_nbours > val_here): # "Greater than" because argrelmax finds peaks
                local_mins.append(i)
                
        self.local_mins = local_mins
            
    def plot_contour(self): # Static or class method?
        
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

#        
#    def plot_slice():
#        
#        
#class landscape_compare:

