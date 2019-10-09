# Entropica QAOA

A modular package providing multiple features and workflow tools for the quantum approximate optimisation algorithm (QAOA), facilitating its use, prototyping, and testing. Includes several different parametrisations, integration with data science and graph analysis libraries such as Pandas and NetworkX, numerous utility functions, and convenient optimiser logging and analysis tools. Documentation contains extensive and didactic examples. 

Read more about EntropicaQAOA on our [blog](https://medium.com/@entropicalabs/entropica-releases-qaoa-package-for-rigettis-quantum-cloud-service-12ea71019436).

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

We assume that the user has already installed Rigetti's pyQuil package, as well as the Rigetti QVM and Quil Compiler. For instructions on how to do so, see Rigetti's documentation [here](http://docs.rigetti.com/en/stable/start.html).

You can install the `EntropicaQAOA` package using [pip](#https://pip.pypa.io/en/stable/quickstart/):

```bash
pip install entropica-qaoa
```
To upgrade to the latest version: 

```bash
pip install --upgrade entropica-qaoa
```

If you want to run the Demo Notebooks, you will additionally need to install `scikit-learn` and `scikit-optimize`, which can be done as follows:

```bash
pip install scikit-learn && pip install scikit-optimize
```

Alternatively, you can clone directly from GitHub:

```bash
git clone https://github.com/entropicalabs/entropica_qaoa.git
```

### Testing

All software tests are located in `entropica_qaoa/tests/`. To run them you will need to install [pytest](https://docs.pytest.org/en/latest/). To speed up the testing, we have tagged tests that require more computational time (~ 5 mins or so)  with `runslow`, and the tests of the notebooks with `notebooks`. The commands are as follows:

 - `pytest` runs the default tests, and skips both the longer tests that need heavier simulations, as well as tests of the Notebooks in the `examples` directory.
 - `pytest --runslow` runs the the tests that require longer time.
 - `pytest --notebooks` runs the Notebook tests. To achieve this, the notebooks are
    converted to python scripts, and then executed. Should any errors occur, this means that the line numbers given in the error
    messages refer to the lines in `<TheNotebook>.py`, and not in
    `<TheNotebook>.ipynb`.
 - `pytest --all` runs all of the above tests. 

### QPU access

EntropicaQAOA provides full native support for Rigetti’s QVM and QPUs. For access to the QPUs, sign up online at https://qcs.rigetti.com/, or reach out to support@rigetti.com.

## Contributing and feedback

If you find any bugs or errors, have feature requests, or code you would like to contribute, feel free to open an issue or send us a pull request on GitHub .

We are always interested to hear about projects built with EntropicaQAOA. If you have an application you’d like to tell us about, drop us an email at devteam@entropicalabs.com.
