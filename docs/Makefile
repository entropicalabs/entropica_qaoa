# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


# compile the notebooks

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


html: source/notebooks/4_CostFunctionsAndVQE.rst\
	  source/notebooks/2_ParameterClasses.rst\
	  source/notebooks/3_AdvancedParameterClasses.rst\
 	  source/notebooks/5_QAOAUtilities.rst\
 	  source/notebooks/1_AnExampleWorkflow.rst\
 	  source/notebooks/6_ClusteringWithQAOA.rst\
 	  source/changelog.rst
	sphinx-build -M html source build


source/changelog.rst: ../CHANGELOG.md
	pandoc --from markdown --to rst -o source/changelog.rst ../CHANGELOG.md


source/notebooks/4_CostFunctionsAndVQE.rst: ../examples/4_CostFunctionsAndVQE.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/4_CostFunctionsAndVQE.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/4_CostFunctionsAndVQE.rst \
	4-CostFunctionsAndVQE

source/notebooks/2_ParameterClasses.rst: ../examples/2_ParameterClasses.ipynb\

	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/2_ParameterClasses.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/2_ParameterClasses.rst \
	2-ParameterClasses

source/notebooks/3_AdvancedParameterClasses.rst: ../examples/3_AdvancedParameterClasses.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/3_AdvancedParameterClasses.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/3_AdvancedParameterClasses.rst \
	3-AdvancedParameterClasses

source/notebooks/1_AnExampleWorkflow.rst: ../examples/1_AnExampleWorkflow.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/1_AnExampleWorkflow.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/1_AnExampleWorkflow.rst \
	1-AnExampleWorkflow

source/notebooks/5_QAOAUtilities.rst: ../examples/5_QAOAUtilities.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/5_QAOAUtilities.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/5_QAOAUtilities.rst \
	5-QAOAUtilities

source/notebooks/6_ClusteringWithQAOA.rst: ../examples/6_ClusteringWithQAOA.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/6_ClusteringWithQAOA.ipynb --output-dir=./source/notebooks/
	./link_converter.sh source/notebooks/6_ClusteringWithQAOA.rst \
	6-ClusteringWithQAOA


clean:
	sphinx-build -M clean source build
