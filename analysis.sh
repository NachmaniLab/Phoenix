#!/bin/bash

source .venv/bin/activate


# human hematopoiesis
python run.py \
    --expression Experiments/human_hematopoiesis/input/expression.csv \
    --cell_types Experiments/human_hematopoiesis/input/cell_types.csv \
    --pseudotime Experiments/human_hematopoiesis/input/pseudotime.csv \
    --reduction Experiments/human_hematopoiesis/input/reduction.csv \
    --output Experiments/human_hematopoiesis/output \
    --organism human \
    --pathway_database msigdb \
    --processes 60 \
    --time 7 \
    --background_mode real


# mouse hematopoiesis
python run.py \
    --expression Experiments/mouse_hematopoiesis/input/expression.csv \
    --cell_types Experiments/mouse_hematopoiesis/input/cell_types.csv \
    --pseudotime Experiments/mouse_hematopoiesis/input/pseudotime.csv \
    --reduction Experiments/mouse_hematopoiesis/input/reduction.csv \
    --output Experiments/mouse_hematopoiesis/output \
    --organism mouse \
    --pathway_database msigdb \
    --processes 60 \
    --time 2 \
    --background_mode real


# zebrafish development
python run.py \
    --expression Experiments/zebrafish_development/input/expression.csv \
    --cell_types Experiments/zebrafish_development/input/cell_types.csv \
    --pseudotime Experiments/zebrafish_development/input/pseudotime.csv \
    --reduction Experiments/zebrafish_development/input/reduction.csv \
    --output Experiments/zebrafish_development/output \
    --organism zebrafish \
    --pathway_database go kegg \
    --processes 60 \
    --time 5 \
    --background_mode real
