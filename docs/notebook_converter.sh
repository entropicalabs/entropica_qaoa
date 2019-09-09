#!/bin/bash

# Simple script to convert Notebooks from ipynb to rst

echo 'converting notebook' $1 to 'rst'
# REMOVECELLS=$"--TagRemovePreprocessor.remove_input_tags='{\"hide_input\", \"hide_all\"}' --TagRemovePreprocessor.remove_all_outputs_tags='{\"hide_output\", \"hide_all\"}'"
OUTDIR='--output-dir=./source/notebooks'
OUTFORMAT='--to rst'

jupyter nbconvert  --TagRemovePreprocessor.remove_input_tags='{"hide_input", "hide_all"}' --TagRemovePreprocessor.remove_all_outputs_tags='{"hide_output", "hide_all"}'  $OUTFORMAT $1 $OUTDIR
