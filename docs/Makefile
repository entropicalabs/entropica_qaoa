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

html: build/html/VQEDemo.html build/html/QAOAParameterDemo.html build/html/AdvancedQAOAParameterDemo.html build/html/UtilitiesDemo.html build/html/QAOAWorkflowDemo.html
	sphinx-build -M html source build


build/html/VQEDemo.html: ../examples/VQEDemo.ipynb
	jupyter nbconvert --to html ../examples/VQEDemo.ipynb --output-dir=./build/html/

build/html/QAOAParameterDemo.html: ../examples/QAOAParameterDemo.ipynb
	jupyter nbconvert --to html ../examples/QAOAParameterDemo.ipynb --output-dir=./build/html/

build/html/AdvancedQAOAParameterDemo.html: ../examples/AdvancedQAOAParameterDemo.ipynb
	jupyter nbconvert --to html ../examples/AdvancedQAOAParameterDemo.ipynb --output-dir=./build/html/

build/html/QAOAWorkflowDemo.html: ../examples/QAOAWorkflowDemo.ipynb
	jupyter nbconvert --to html ../examples/QAOAWorkflowDemo.ipynb --output-dir=./build/html/

build/html/UtilitiesDemo.html: ../examples/UtilitiesDemo.ipynb
	jupyter nbconvert --to html ../examples/UtilitiesDemo.ipynb --output-dir=./build/html/

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile build/html/VQEDemo.ipynb
# 	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# build/html/VQEDemo.html: ../examples/VQEDemo.ipynb
#	jupyter nbconvert --to html ../examples/VQEDemo.ipynb --output-dir=./build/html/
