# entropica_qaoa
A package implementing the quantum approximate optimisation algorithm (QAOA), providing a number of different features, parametrisations, and utility functions. 

## Documentation
The documentation can be installed and built by following these two steps:

**Install the Prerequisites**
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints nbsphinx
```
**Compile the documentation**
```bash
cd docs && make html
```

## Installation

We assume that the user has already installed Rigetti's pyQuil package, as well as the Rigetti QVM and Quil Compiler. For instructions on how to do so, see the Rigetti documentation here: http://docs.rigetti.com/en/stable/start.html

Installation of the `entropica_qaoa` package can be performed in a few simple steps.

1. Open terminal and enter the site-packages folder of your preferred Python environment.

For those with Anaconda installed, the command looks like:
```
cd /anaconda3/envs/<my-env>/lib/pythonX.Y/site-packages/
```
For those unsure of the location of their site-packages folder, you can simply run 'pip show <package name>' and your terminal will display the directory location of your python packages.

2. Clone the repository into your site-packages folder:

```
git clone https://github.com/entropicalabs/entropica_qaoa.git 
```
3. Move into the entropica_qaoa directory and install the package:

```bash
cd entropica_qaoa
python setup.py install
```

You can now import this package as you would any conda- or pip-installed library!

## Testing
 - `pytest` runs the default tests and skips longer tests that need heavy simulations and tests of the Notebooks in `examples/`
 - `pytest --runslow` the tests with heavy simulations. I didn't fix the seed for the random generator yet, so sometimes the test doesn't even work.... If it takes longer than 5 Minutes just kill the tests and restart it.
 - `pytest --notebooks` runs the Notebook tests. To achieve this, the notebooks are converted to python scripts which are then executed. So the line numbers in the error messages refer to the lines in `<TheNotebook>.py` and not in `<TheNotebook>.ipynb`.
 - `pytest --all` runs all of the above tests. Ideally all of them pass, before you do a `git push`
 - If you need more infos than `pytest` give you be default: Use the toggle `pytest (options) -s` to get all output.
 - with `pytest tests/<testfile>` single tests can be run to check single modules.

## Contributing
This project is hosted on GitHub and can be found at https://github.com/entropicalabs/entropica_qaoa.git. If you have feature requests or have already implemented them, feel free to send us a pull request. 
