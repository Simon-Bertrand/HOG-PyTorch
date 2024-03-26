#!/bin/bash

# Vérifier si nbconvert est installé, sinon l'installer
if ! command -v jupyter nbconvert &> /dev/null
then
    pip install nbconvert jupyter
fi

# Convertir le notebook en markdown
jupyter nbconvert --to markdown README.ipynb --output README

# Renommer le dossier README_files en figs
rm -rf figs && mv README_files figs

# Remplacer les occurrences de README_files/ par figs/ dans README.md
sed -i 's/README_files\//figs\//g' README.md