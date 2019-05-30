Here some of the things I noticed, while reading through `Utilities Demo.ipynb`
 - Docstrings:
   I really hate to be that person, but it would be great to have all the docstrings
   in [numpy style](https://numpydoc.readthedocs.io/en/latest/format.html). Not
   because I am anal and want to annoy you, but because this enables
   [docstring inheritance](https://pypi.org/project/custom_inherit/) in qaoa.paramaters
   _and_ automatic documentation generation with sphinx.
 - `distances_dataset()`:
   The keys in the returned dict are gonna become non-unique once you have
   more than 10 datapoints (because `f"{12}{3}" == f"{1}{23}"`).
   `tmp_dict` in line 179 is not neccesary. You can update a dict by simply doing
   my_dict[key] = val
   *Suggestion*: Replace lines 179 and 180 with
   `distances[(i, j)] = np.linalg.norm(data[i] - data[j])`.
   *Another suggestion*: If you care about performance and memory: Store the
   distances in a matrix where the (i,j)-th entry is the distance between the i-th
   and j-th data point. I am sure there is an elegant one-linear to create the this
   distance matrix with numpy. (so the matrix is symmetric and the diagonal 0)
   This means, you will also have to update `hamiltonian_from_dict`, but the changes
   should be minor.
 - `create_gaussian_2Dclusters`:
   shorten the parameter list by passing a list of 
   covariance matrices instead of variances and covariances seperately?
 - all the graph stuff:
   change naming convention from `hamiltonian_from_networkx`
   to `hamiltonian_from_graph` because the it is important, that we come from a graph,
   not what the name of the used library is? (so replace `networkx` with `graph` in
   all function names)
 - `plot_networkx_graph(G)`:
   Can't you just use the node attribute "weight" to specify
   the bias? (see https://networkx.github.io/documentation/networkx-2.1/reference/classes/generated/networkx.Graph.nodes.html)
   In `qaoa.paramaters.AbstractQAOAParameters.plot()` there is an optional argument
   `ax=None` where the user can pass matplotlib.axes to plot on. This is useful, if
   you want to plot the parameters only as part of a subplot in a bigger layout.
   Maybe add this option here as well?
 - `create_random_hamiltonian`: 
   Yup, `pair_terms` is probably not very useful and the other suggestions are more
   likely to be used. Go crazy with the kwargs I would say.
 - `networkx_from_hamiltonian`:
   currently doesn't take in a hamiltonian, but vertex_pairs and edge_weights...
   Either change the parameters or the name, but the current situation is a bit
   confusing.
 - `plot_cluster_data()`:
   Same with as for `plot_networkx_graph()`: Add an kwarg `ax=None`.
