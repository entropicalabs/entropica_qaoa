# Entropica QAOA

A package implementing the Quantum Approximate Optimisation Algorithm (QAOA), providing a number of different features, parametrisations, and utility functions. 


## Documentation

The documentation can be found [here](https://docs.entropicalabs.io/qaoa/).

Alternatively you can compile it yourself by following the instructions below:

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

If you don't have them already, install first Rigetti's pyQquil package and their QVM and Quil Compiler. For instructions on how to do so, see the Rigetti documentation here: http://docs.rigetti.com/en/stable/start.html.

In a Python3.6+ virtual environment you can install the `entropica_qaoa`  package using [pip](#https://pip.pypa.io/en/stable/quickstart/)

```bash
pip install entropica_qaoa
```
and if you have it already installed upgraded via

```bash
pip install --upgrade entropica_qaoa
```

If you want to run the Demo Notebooks you will additionally need `scikit-learn` and `scikit-optimize` which can also be installed using pip:

```bash
pip install scikit-learn && pip install scikit-optimize
```


## Development and Contributing

This project is hosted at [github](https://github.com/entropicalabs/entropica_qaoa) and can be cloned using

```bash
git clone https://github.com/entropicalabs/entropica_qaoa.git
```

If you have feature requests or already implemented them, feel free to open an issue or send us a pull request.

### Testing

All tests are located in `entropica_qaoa/tests/`. To run them you will need [pytest](https://docs.pytest.org/en/latest/) installed. To run all tests, including the notebook tests, you will additionally need to have [nbconvert](https://github.com/jupyter/nbconvert) and [scikit-learn](https://scikit-learn.org/stable/) and [scikit-optimize](https://scikit-optimize.github.io/) installed.

To speed up the testing, we have tagged tests that take a lot of time with `runslow` and the tests of the notebooks with `notebooks`. This means that a bare ```pytest``` doesn't run those tests. More below.

 - `pytest` runs the default tests and skips longer tests that need heavy simulations and tests of the Notebooks in `examples/`
 - `pytest --runslow` the tests with heavy simulations. I didn't fix the seed for the
   random generator yet, so sometimes the test doesn't even work.... If it takes longer than
   5 Minutes just kill the tests and restart it.
 - `pytest --notebooks` runs the Notebook tests. To achieve this, the notebooks are
    converted to python scripts which are then executed. So the line numbers in the error
    messages refer to the lines in `<TheNotebook>.py` and not in
    `<TheNotebook>.ipynb`.
 - `pytest --all` runs all of the above tests. Ideally all of them pass, before you do a
    `git push`
 - If you need more infos than `pytest` give you be default: Use the toggle
    `pytest (options) -s` to get all output.
 - With `pytest tests/<testfile>` single tests can be run to check single modules.
