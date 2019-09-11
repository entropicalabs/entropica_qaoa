# Entropica QAOA

A package implementing the Quantum Approximate Optimisation Algorithm (QAOA), providing a number of different features, parametrisations, and utility functions. 


## Documentation

The documentation for EntropicaQAOA can be found [here](https://docs.entropicalabs.io/qaoa/). Alternatively, it can be complied locally as follows:

**Install the Prerequisites**
```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints nbsphinx nbconvert
```

**Compile the documentation**
```bash
cd docs && make html
```

The compiled HTML version of the documentation is then found in
`entropica_qaoa/docs/build/html`.


## Installation

We assume that the user has already installed Rigetti's pyQuil package, as well as the Rigetti QVM and Quil Compiler. For instructions on how to do so, see the Rigetti documentation [here](http://docs.rigetti.com/en/stable/start.html).

You can install the `entropica_qaoa` package using [pip](#https://pip.pypa.io/en/stable/quickstart/):

```bash
pip install entropica_qaoa
```
To upgrade to the latest version: 

```bash
pip install --upgrade entropica_qaoa
```

If you want to run the Demo Notebooks, you will additionally need to install `scikit-learn` and `scikit-optimize`, which can be done as follows:

```bash
pip install scikit-learn && pip install scikit-optimize
```

### Testing

All software tests are located in `entropica_qaoa/tests/`. To run them you will need to install [pytest](https://docs.pytest.org/en/latest/). To speed up the testing, we have tagged tests that require more computational time (~ 5 mins or so)  with `runslow`, and the tests of the notebooks with `notebooks`. This means that a bare $

 - `pytest` runs the default tests, and skips both the longer tests that need heavier simulations, as well as tests of the Notebooks in the `examples` directory.
 - `pytest --runslow` runs the the tests that require longer time.
 - `pytest --notebooks` runs the Notebook tests. To achieve this, the notebooks are
    converted to python scripts, and then executed. Should any errors occur, this means that the line numbers given in the error
    messages refer to the lines in `<TheNotebook>.py`, and not in
    `<TheNotebook>.ipynb`.
 - `pytest --all` runs all of the above tests. 

## Contributing and feedback

This project is hosted on GitHub, and can be cloned as follows:

```bash
git clone https://github.com/entropicalabs/entropica_qaoa.git
```

If you have feature requests, or have already implemented them, feel free to open an issue or send us a pull request.

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you’d like to tell us about, drop us an email at devteam@entropicalabs.com.
