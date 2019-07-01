# Empirically test the local_minima function in landscapes.py

import numpy as np
import scipy.signal as signal
import itertools

def get_local_minima(cost_array):
    
    """
    Searches through an array to find locations of minima
    Since there may be multiple minima along any direction, a log(N) binary search cannot be used.
    This method exhaustively walks through element by element, comparing to neighbours
    """
    
    local_mins = []
    
    dims = np.shape(cost_array)
    inds_ranges = dims - np.ones(len(dims),)
    n_els = np.prod(dims)    
    
    iterables = [[-1,0,1]]*len(dims)
    neighb_array = [np.array(t) for t in itertools.product(*iterables)]
    n_neighbs = len(neighb_array)
    
    neighs_above = np.zeros(dims)
    for i in range(n_els):
        
        inds = np.unravel_index(i, tuple(dims))
        val_here = cost_array[tuple(inds)]
        
        neighb_vals = np.zeros((n_neighbs,))
        for j in range(n_neighbs):
            
            neighb_inds = inds + neighb_array[j]
            
            above_range = np.nonzero(neighb_inds > inds_ranges)
            below_range = np.nonzero(neighb_inds < 0)
            out_of_range = np.hstack((above_range,below_range))
            
            if len(out_of_range[0]) > 0 :
                continue
            
            neighb_vals[j] = cost_array[tuple(neighb_inds)]
        
        larger_neighbs = sum(neighb_vals > val_here) 

        if larger_neighbs == n_neighbs-1:
            local_mins.append(inds) # Definite minimum
        
        neighs_above[tuple(inds)] = larger_neighbs

    return local_mins, neighs_above
            
            

def generate_rand_min_matr(n):
    
    matr = np.ones((n,n))
    n_min = np.random.randint(low=1,high=int(n/2))
    
    coords = np.random.randint(low=1,high=n-2,size=(n_min,2))
    #print('raw = ', coords)
    coords = np.unique(coords, axis=0)
    #print('clean =', coords)
    
    for i in range(len(coords)):
    
        matr[tuple(coords[i])] = -0.5
        
    return matr, coords

#matr, coords = generate_rand_min_matr(10)
#local_mins,neighbs_above = get_local_minima(matr)

#print(matr)
#print(coords)
#print(local_mins)
#print(neighbs_above)
#
    
the_coords = []
found_coords = []
cases = 0
for i in range(100):
    
    matr, coords = generate_rand_min_matr(10)
    local_mins,_ = get_local_minima(matr)
    
    return_same = np.array_equal(np.array(local_mins),coords)
    
    if len(local_mins) > 0 and not return_same:
        cases += 1
        the_coords.append(coords)
        found_coords.append(local_mins)

#    def local_minima(self): # Static or class method?
#        
#        """
#        Finds the list of local minima in the desired array
#        signal.argrelmax finds maxima, so we find the maxima of -1*array
#        signal.argrelmax returns any kind of minima (eg along any of the directions), not necessarily a minima in all directions
#        we thus further analyse the result to obtain those points that are true local minima in all directions
#        note that this method ignores points at the extremes of a given dimension
#        
#        TODO: systematically ensure this method is correct by constructing landscapes with known local minima & make sure it returns right result
#        """
#         
#        mins = signal.argrelmax(-1*self.func_vals)
#        mins = [[mins[0][i],mins[1][i]] for i in range(len(mins[0]))]
#        
#        local_mins = []
#        for i in mins: 
#            
#            val_here = self.func_vals[tuple(i)] # Function value at coordinate i
#            
#            # Get indices of neighbouring points, checking if the point on the boundary of the coordinate system (ignore if so)
#            edge_pts = [j for j in range(len(i)) if i[j] == self.shape[j]-1 or i[j] == 0]
#            if edge_pts: 
#                continue
#            else:
#                nbours = np.array([i]*4) + np.array([[-1,0], [0,-1], [1, 0], [0,1]]) # coordinates of neighbouring points (NEED TO EXTEND TO ARBITRARY DIMS)
#            
#            val_nbours = [self.func_vals[tuple(j)] for j in nbours] # Function values at neighbouring points
#
#            if all(val_nbours > val_here): # "Greater than" because argrelmax finds peaks
#                local_mins.append(i)
#                
#        self.local_mins = local_mins

#def local_minima(func_vals): # Static or class method?
#        
#        """
#        Finds the list of local minima in the desired array
#        signal.argrelmin finds maxima, so we find the maxima of -1*array
#        signal.argrelmax returns any kind of minima (eg along any of the directions), not necessarily a minima in all directions
#        we thus further analyse the result to obtain those points that are true local minima in all directions
#        note that this method ignores points at the extremes of a given dimension
#        
#        TODO: systematically ensure this method is correct by constructing landscapes with known local minima & make sure it returns right result
#        """
#        func_shape = np.shape(func_vals)
#         
#        mins = signal.argrelmin(func_vals)
#        mins = [[mins[0][i],mins[1][i]] for i in range(len(mins[0]))]
#        
#        print('mins found at', mins)
#        
#        local_mins = []
#        for i in mins: 
#            
#            val_here = func_vals[tuple(i)] # Function value at coordinate i
#            
#            # Get indices of neighbouring points, checking if the point on the boundary of the coordinate system (ignore if so)
#            edge_pts = [j for j in range(len(i)) if i[j] == func_shape[j]-1 or i[j] == 0]
#            if edge_pts: 
#                continue
#            else:
#                nbours = np.array([i]*4) + np.array([[-1,0], [0,-1], [1, 0], [0,1]]) # coordinates of neighbouring points (NEED TO EXTEND TO ARBITRARY DIMS)
#            
#            val_nbours = [func_vals[tuple(j)] for j in nbours] # Function values at neighbouring points
#
#            if all(val_nbours >= val_here): 
#                local_mins.append(i)
#               
#        print('real mins determined to be',local_mins)
#        
#        return np.unique(np.array(local_mins,dtype=int),axis=0)
    
    
#a = [5,4,3,2,1,1,1,1,2,3,4,5]
#mins = minima_locate(a)
#print(mins)



#vect = [1,1,2,3,2,1,0,1,2,3]
#inds = peak_find(vect)
#print(inds)

#def LR_check(vec,inds):
#
#    if vec[inds[0]] >= vec[inds[0]-1] and vec[inds[1]] >= vec[inds[1]+1]:
#        new_inds = [inds[0]-1,inds[1]+1]
#        return LR_check(vec,new_inds)  
#    else:
#        return inds
#
#def minima_locate(vec,true_inds):
#    
#    """
#    FUNCTION ONLY GOOD FOR A SINGLE PEAK :(
#    Function to find minima in a 1D array
#    Also returns the indices corresponding to flat regions that are minima 'plateau' (none of Scipy's standard functions do this)
#    Warning: Doesn't consider the case of odd-length lists
#    """
#    
#    inds = []
#    
#    if len(vec) == 2:
#        return inds
#    elif len(vec) == 3 and vec[1] < vec[0] and vec[1] < vec[2]:
#        inds.append()
#
#    middle_ind = [int(len(vec)/2-1),int(len(vec)/2)]
#    print(middle_ind)
#    
#    if middle_ind[0] == 0 or middle_ind[1] == len(vec):
#        print('ends')
#        return inds
#    
#    else:
#        val_left = vec[middle_ind[0]]
#        val_right = vec[middle_ind[1]]
#        print(middle_ind, val_left,val_right)
#        
#        if val_left < val_right and val_left < vec[middle_ind[0]-1]:
#            inds.append(middle_ind[0]) 
#        elif val_right < val_left and val_right < vec[middle_ind[1]+1]:
#            inds.append(middle_ind[1])
#        elif val_left == val_right:
#            flat_inds = LR_check(vec,middle_ind)
#            inds.append(flat_inds)
#            print('LRcheck', flat_inds, inds)
#            
#        left_vec = vec[:middle_ind[1]]
#        right_vec = vec[middle_ind[1]:] 
#        l_inds = np.arange(middle_ind[1])
#        r_inds = np.arange(middle_ind[1],len(vec))
#        
#        print('Rightside')
#        inds.append(minima_locate(right_vec,l_inds))
#        print('Leftside')
#        inds.append(minima_locate(left_vec,r_inds))