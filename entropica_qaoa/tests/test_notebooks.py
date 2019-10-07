"""
Tests to run the notebooks. They work by converting the notebooks
to a python script via nbconvert and then running the resulting .py file.
"""

import subprocess
import pytest
import importlib

import sys, os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')


def notebook_test_function(name):
    command = f"jupyter nbconvert --to script {name} --output ../entropica_qaoa/tests/converted"

    print(f"Converting {name} to a python script")
    ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

    if ret_code is not 0:
        print(f"The command '{command}' failed. "
              "Run it manually, to see what went wrong.")
    import converted
    importlib.reload(converted)


@pytest.mark.notebook
def test_1_AnExampleWorkflow():
    notebook_test_function("../examples/1_AnExampleWorkflow.ipynb")


@pytest.mark.notebook
def test_2_ParameterClasses():
    notebook_test_function("../examples/2_ParameterClasses.ipynb")


@pytest.mark.notebook
def test_3_AdvancedParameterClasses():
    notebook_test_function("../examples/3_AdvancedParameterClasses.ipynb")


@pytest.mark.notebook
def test_4_CostFunctionsAndVQE():
    notebook_test_function("../examples/4_CostFunctionsAndVQE.ipynb")

@pytest.mark.notebook
def test_5_QAOAUtilities():
    notebook_test_function("../examples/5_QAOAUtilities.ipynb")

@pytest.mark.notebook
def test_6_QUBOwithQAOA():
    notebook_test_function("../examples/6_SolvingQUBOwithQAOA.ipynb")


@pytest.mark.notebook
def test_7_ClusteringWithQAOA():
    notebook_test_function("../examples/7_ClusteringWithQAOA.ipynb")



# @pytest.mark.notebook
# def test_1_AnExampleWorkflow():
#     name = "../examples/1_AnExampleWorkflow.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.1_AnExampleWorkflow


# @pytest.mark.notebook
# def test_6_ClusteringWithQAOA():
#     name = "../examples/6_ClusteringWithQAOA.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.6_ClusteringWithQAOA


# @pytest.mark.notebook
# def test_2_ParameterClasses():
#     name = "../examples/2_ParameterClasses.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.2_ParameterClasses


# @pytest.mark.notebook
# def test_5_CostFunctionAndVQE():
#     name = "../examples/5_CostFunctionAndVQE.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.5_CostFunctionAndVQE


# @pytest.mark.notebook
# def test_3_AdvancedParameterClasses():
#     name = "../examples/3_AdvancedParameterClasses.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.3_AdvancedParameterClasses


# @pytest.mark.notebook
# def test_4_CostFunctionsAndVQE():
#     name = "../examples/4_CostFunctionsAndVQE.ipynb"
#     command = f"jupyter nbconvert --to script {name}"

#     print(f"Converting {name} to a python script")
#     ret_code = subprocess.call(command.split(" "), stderr=None, stdout=None)

#     if ret_code is not 0:
#         print(f"The command '{command}' failed. "
#               "Run it manually, to see what went wrong.")
#     import examples.4_CostFunctionsAndVQE
