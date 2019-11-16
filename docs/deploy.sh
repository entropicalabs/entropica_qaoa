#!/bin/bash
rsync -av build/html/. dist/qaoa/.
git add dist/qaoa/.
