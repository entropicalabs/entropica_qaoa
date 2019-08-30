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


html: VQEDemo.rst QAOAParameterDemo.rst AdvancedQAOAParameterDemo.rst\
 	  UtilitiesDemo.rst QAOAWorkflowDemo.rst
	sphinx-build -M html source build


VQEDemo.rst: ../examples/VQEDemo.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/VQEDemo.ipynb --output-dir=./source/notebooks/

QAOAParameterDemo.rst: ../examples/QAOAParameterDemo.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/QAOAParameterDemo.ipynb --output-dir=./source/notebooks/

AdvancedQAOAParameterDemo.rst: ../examples/AdvancedQAOAParameterDemo.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/AdvancedQAOAParameterDemo.ipynb --output-dir=./source/notebooks/

QAOAWorkflowDemo.rst: ../examples/QAOAWorkflowDemo.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/QAOAWorkflowDemo.ipynb --output-dir=./source/notebooks/

UtilitiesDemo.rst: ../examples/UtilitiesDemo.ipynb
	jupyter nbconvert \
	--TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' \
	--TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}' \
	--to rst ../examples/UtilitiesDemo.ipynb --output-dir=./source/notebooks/

clean:
	sphinx-build -M clean source build
