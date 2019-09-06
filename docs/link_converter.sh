#!/bin/bash

echo 'converting links in' $1
sed -i "1i .. _$2:" $1
sed -i 's/``1_AnExampleWorkflow.ipynb``/:ref:`1-AnExampleWorkflow`/g' $1
sed -i 's/``2_ParameterClasses.ipynb``/:ref:`2-ParameterClasses`/g' $1
sed -i 's/``3_AdvancedParameterClasses.ipynb``/:ref:`3-AdvancedParameterClasses`/g' $1
sed -i 's/``4_CostFunctionsAndVQE.ipynb``/:ref:`4-CostFunctionsAndVQE`/g' $1
sed -i 's/``5_QAOAUtilities.ipynb``/:ref:`5-QAOAUtilities`/g' $1
sed -i 's/``6_ClusteringWithQAOA.ipynb``/:ref:`6-ClusteringWithQAOA`/g' $1

