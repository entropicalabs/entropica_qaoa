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
%: Makefile
	jupyter nbconvert --to html ../examples/VQEDemo.ipynb --output-dir=./build/
	jupyter nbconvert --to html ../examples/QAOAParameterDemo.ipynb --output-dir=./build/
	jupyter nbconvert --to html ../examples/QAOAWorkflowDemo.ipynb --output-dir=./build/
	jupyter nbconvert --to html ../examples/AdvancedQAOAParameterDemo.ipynb --output-dir=./build/
	jupyter nbconvert --to html ../examples/UtilitiesDemo.ipynb --output-dir=./build/
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

