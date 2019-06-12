import numpy as np
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
        
        # Have a nice str or repr method for the class attributes

#    def landscape_statistics():
        
        # IDEAS: 
        #   Peak finding algorithms: find peaks & troughs along all the different dimensions, then compare where there is overlap?
        #   A Scipy method: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.argrelextrema.html
        #   Plotly: https://plot.ly/python/peak-finding/
        

    def plot_contour(self):
        
        assert (len(self.the_parameters) == 2), "Two parameters required for visualisation on 3D plot."
        
        fig, ax = plt.subplots()
        
        param1, param2 = np.meshgrid(self.parameter_ranges[0],self.parameter_ranges[1],indexing='ij')
        
        contour_ = ax.contour(param1,param2,self.func_vals)
        ax.clabel(contour_, inline=1, fontsize=10)
        
    def plot_3D(self):
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

