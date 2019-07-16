"""
Tests to run the notebooks. They work by converting the notebooks
to a python script via nbconvert and then running the resulting .py file.
"""

import subprocess
import pytest


@pytest.mark.notebook
def test_VQEDemo():
    name = "examples/VQEDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.VQEDemo


@pytest.mark.notebook
def test_MathIntuition_notebook():
    name = "examples/The_Mathematical_Intuition_of_QAOA.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.The_Mathematical_Intuition_of_QAOA


@pytest.mark.notebook
def test_LandscapesDemo_notebook():
    name = "examples/LandscapesDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.LandscapesDemo


@pytest.mark.notebook
def test_QAOAClusteringDemo_notebook():
    name = "examples/QAOAClusteringDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.QAOAClusteringDemo


@pytest.mark.notebook
def test_QAOAParameterDemo():
    name = "examples/QAOAParameterDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.QAOAParameterDemo


@pytest.mark.notebook
def test_QAOAWorkflowDemo():
    name = "examples/QAOAWorkflowDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.QAOAWorkflowDemo


@pytest.mark.notebook
def test_UtilitiesDemo():
    name = "examples/UtilitiesDemo.ipynb"
    command = f"jupyter nbconvert --to script {name}"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import forest_qaoa.examples.UtilitiesDemo
