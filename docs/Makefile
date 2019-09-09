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

# Don't know what this does, but it was already there
.PHONY: help Makefile


# This was a smart catch-all target that enabled _all_ targets like
# `make html`, `make pdf`, `make latex` etc. It doesn't work anymore,
# because we added all the extra dependencies like the Notebook conversion
# and Changelog conversion.

# %: Makefile
# 	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# main target to create the html docs. To create PDF or Latex docs
# copy this target and replace `html` with `pdf` / `latex` where neccesary.
html: source/notebooks/4_CostFunctionsAndVQE.rst\
	  source/notebooks/2_ParameterClasses.rst\
	  source/notebooks/3_AdvancedParameterClasses.rst\
 	  source/notebooks/5_QAOAUtilities.rst\
 	  source/notebooks/1_AnExampleWorkflow.rst\
 	  source/notebooks/6_ClusteringWithQAOA.rst\
 	  source/changelog.rst
	sphinx-build -M html source build


# convert the changelog from .md to .rst and save it /docs/source to include
# it in the html documentation.
source/changelog.rst: ../CHANGELOG.md
	pandoc --from markdown --to rst -o source/changelog.rst ../CHANGELOG.md

# convert the notebooks from .ipynb to .md and then place tags to them in the
# first line, s.t. that we can link to them.
# Conversion to them happens in ./notebook_converter.sh
# Tag placement and link replacement in ./link_converter.sh
source/notebooks/4_CostFunctionsAndVQE.rst: ../examples/4_CostFunctionsAndVQE.ipynb
	./notebook_converter.sh ../examples/4_CostFunctionsAndVQE.ipynb
	./link_converter.sh source/notebooks/4_CostFunctionsAndVQE.rst \
	4-CostFunctionsAndVQE

source/notebooks/2_ParameterClasses.rst: ../examples/2_ParameterClasses.ipynb
	./notebook_converter.sh ../examples/2_ParameterClasses.ipynb
	./link_converter.sh source/notebooks/2_ParameterClasses.rst \
	2-ParameterClasses

source/notebooks/3_AdvancedParameterClasses.rst: ../examples/3_AdvancedParameterClasses.ipynb
	./notebook_converter.sh ../examples/3_AdvancedParameterClasses.ipynb
	./link_converter.sh source/notebooks/3_AdvancedParameterClasses.rst \
	3-AdvancedParameterClasses

source/notebooks/1_AnExampleWorkflow.rst: ../examples/1_AnExampleWorkflow.ipynb
	./notebook_converter.sh ../examples/1_AnExampleWorkflow.ipynb
	./link_converter.sh source/notebooks/1_AnExampleWorkflow.rst \
	1-AnExampleWorkflow

source/notebooks/5_QAOAUtilities.rst: ../examples/5_QAOAUtilities.ipynb
	./notebook_converter.sh ../examples/5_QAOAUtilities.ipynb
	./link_converter.sh source/notebooks/5_QAOAUtilities.rst \
	5-QAOAUtilities

source/notebooks/6_ClusteringWithQAOA.rst: ../examples/6_ClusteringWithQAOA.ipynb
	./notebook_converter.sh ../examples/6_ClusteringWithQAOA.ipynb
	./link_converter.sh source/notebooks/6_ClusteringWithQAOA.rst \
	6-ClusteringWithQAOA

# lastly the clean target. One should probably add "rm source/notebooks/*"
# and "rm source/changelog.rst"
clean:
	sphinx-build -M clean source build
