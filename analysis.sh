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
    --background_mode real \
    --repeats 150 \
    --set_fraction 1.0 \
    --min_set_size 2 \
    --fdr_threshold 0.01 \
    --corrected_effect_size_threshold 1.2 \
    --importance_lower_threshold 0.05 \
    --importance_gene_fraction_threshold 0.5 \
    --processes 60 \
    --time 7


# mouse hematopoiesis
python run.py \
    --expression Experiments/mouse_hematopoiesis/input/expression.csv \
    --cell_types Experiments/mouse_hematopoiesis/input/cell_types.csv \
    --pseudotime Experiments/mouse_hematopoiesis/input/pseudotime.csv \
    --reduction Experiments/mouse_hematopoiesis/input/reduction.csv \
    --output Experiments/mouse_hematopoiesis/output \
    --organism mouse \
    --pathway_database msigdb \
    --background_mode real \
    --repeats 150 \
    --set_fraction 1.0 \
    --min_set_size 2 \
    --fdr_threshold 0.01 \
    --corrected_effect_size_threshold 1.2 \
    --importance_lower_threshold 0.05 \
    --importance_gene_fraction_threshold 0.5 \
    --processes 60 \
    --time 2 


# zebrafish development
python run.py \
    --expression Experiments/zebrafish_development/input/expression.csv \
    --cell_types Experiments/zebrafish_development/input/cell_types.csv \
    --pseudotime Experiments/zebrafish_development/input/pseudotime.csv \
    --reduction Experiments/zebrafish_development/input/reduction.csv \
    --output Experiments/zebrafish_development/output \
    --organism zebrafish \
    --pathway_database go kegg \
    --background_mode real \
    --repeats 150 \
    --set_fraction 1.0 \
    --min_set_size 2 \
    --fdr_threshold 0.01 \
    --corrected_effect_size_threshold 1.2 \
    --importance_lower_threshold 0.05 \
    --importance_gene_fraction_threshold 0.5 \
    --processes 60 \
    --time 5
