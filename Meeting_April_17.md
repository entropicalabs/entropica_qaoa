17 APRIL:

    TO DISCUSS:

    - Division of labour: for exmaple JL to build out and integrate the parts he has suggested, Ewan to focus on visualisation and landscape stuff? Then see where we get.
    - How to synchronise the two smoothly: where all JL's basic functions are for Ewan to base his code on
    - Go through the comments in the PackageOutline.md file, and Ewan's files.
    - Rough timeline / deadline for the minimal version of the package (& define what that means) to be sent to Rigetti & tested by our team. 
    
    NOTES and ACTION POINTS:
    
    - Ewan to focus on Visualisation.py and Utils.py, JL to focus on everything else
    - Minimal package is essentially everything in PackageOutline.md, minus a few of Ewan's sillier or more ambitious suggestions.
    - JL will also look at adding the ability to have arbitrary cost functions (not just ZZ terms)
    
    - [EWAN] Draw a diagram / hierarchy illustration of the package and all its modules and classes to fully understand its structure and explain it to people.
    - [EWAN] Harmonise code with JL's Pyquil modified package (download from his Github)
    - [EWAN] Landscape should have height for cost, colour for variance.
    - [EWAN] Think more about the plot.object() or plot(object) question.

3 MAY:

    Talking points:
    
    Parameter definition for sweeps
    Logging function values and number of function calls: this is partially implemented in cost_function call methods, but can we extend them?
    Build up more examples of user workflow
    Best type of data structures to use for input data (eg graphs, datasets etc): dictionaries?
    
27 MAY:

    - Does everything run seamlessly on the QPU?
    - Systematically profile the runtime and memory use?
    - Comment out Rigetti's debugging lines that Joaquin found
    - Rename some of the longer classes / methods? 
    - Benchmark "noisy" and illustrate its use. Also show use of "log". [At least we need to check that these work as expected, even if we don't explicitly demo them in notebooks].
    - Print out of parameters for the General, Alternating classes etc. Check that it all makes sense, and is consistent across the classes.
    - Showcase the other parameter classes (adiabatic, Fourier)
    - How would a user easily/cleanly modify the number of timesteps to use without having to re-instantiate the whole claass again?
    - Clean up existing notebooks: add URLs and references, improve explanations, clearer/better naming of variables, keeping imports to a minimum, etc.
    - Option to start at a state other than the all |+> 
    - Rename _qaoa_annealing_program in QAOA cost function file?
    - Showcase the use of non-diagonal (or more general) cost functions? Or just leave as documented, without explicit example. 
