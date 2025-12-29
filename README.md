## Overview

*Phoenix* is a comprehensive **P**at**H**way **O**ntology **EN**r**I**chment tool in single-cell e**X**pression data, focusing on identifying important pathways that distinguish between cell-types and across pseudo-time.
It evaluates biological pathways for their predictive power in two key areas: predicting cell-types (discrete values) and estimating pseudo-time (continuous values), using classification and regression models, respectively.
To assess the significance of the identified pathways, the tool compares the performance of each gene set against a random set of genes of equivalent size.
Gene annotations that significantly outperform random gene sets are considered particularly relevant within the specific context of the data.


## Installation

To install the tool, clone this repository:

```
git clone https://github.com/NachmaniLab/Phoenix.git
cd Phoenix
```

Then, install the required dependencies using either `venv` or `conda`:

### Using `venv`

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Using `conda`

```
conda create prefix .venv python=3.11
conda activate ./.venv
conda install --file requirements.txt
```


## Usage

To run the tool, execute the `run.py` script with the relevant parameters.

### Basic arguments

Provide input data:

* `expression`: Path to single-cell raw expression data where rows represent cells and columns represent gene symbols (CSV file).
* `reduction`: Path to dimensionality reduction coordinates where rows represent cells and columns include names of the first two components (CSV file), or a dimensionality reduction method: `pca`, `umap` or `tsne`. Default: `umap`.

Provide at least one target values:

* `cell_types`: Path to cell-type annotations where rows represent cells and first column presents cell-types (CSV file).
* `pseudotime`: Path to pseudo-time values where rows represent cells and columns include names of different trajectories (CSV file).

Specify a known pathway database for a specific organism, or provide a custom gene set list:

* `organism`: Organism name for pathway annotations.
* `pathway_database`: Known pathway database: `go`, `kegg` or `msigdb`.
* `custom_pathways`: Path to custom gene sets where columns represent set names and rows include gene symbols (CSV file).

Provide output path:

* `output`: Path to output directory.

### Additional arguments

Customize preprocessing parameters:

* `preprocessed`: Whether expression data are log-normalized. Default: `False`.
* `exclude_cell_types`: Cell-types to exclude from the analysis.
* `exclude_lineages`: Lineages to exclude from the analysis.

Customize feature selection parameters:

* `feature_selection`: Feature selection method applied to each gene set: `RF` or `ANOVA`. Default: `RF`.
* `set_fraction`: Fraction of genes to select from each gene set. Default: `0.75`.
* `min_set_size`: Minimum number of genes to select from each gene set. Default: `2`.

Customize prediction model parameters:

* `classifier`: Classification model: `Reg`, `KNN`, `SVM`, `DTree`, `RF`, `LGBM`, `XGB`, `GradBoost` or `MLP`. Default: `RF`.
* `regressor`: Regression model: `Reg`, `KNN`, `SVM`, `DTree`, `RF`, `LGBM`, `XGB`, `GradBoost` or `MLP`. Default: `RF`.
* `classification_metric`: Classification score: `accuracy`, `accuracy_balanced`, `f1`, `f1_weighted`, `f1_macro`, `f1_micro`, `f1_weighted_icf` or `recall_weighted_icf`. Default: `f1_weighted_icf`.
* `regression_metric`: Regression score: `neg_mean_absolute_error`, `neg_mean_squared_error` or `neg_root_mean_squared_error`. Default: `neg_root_mean_squared_error`.
* `cross_validation`: Number of cross-validation folds. Default: `10`.
* `seed`: Seed for reproducibility. Default: `3407`.

Customize background distribution parameters:

* `background_mode`: Strategy for constructing background score distributions used in pathway significance testing. `real` builds backgrounds from scores of real pathways, `random` builds backgrounds from randomly sampled gene sets at predefined sizes, and `auto` chooses automatically based on the size of the gene set database. Default: `auto` (recommended)
* `repeats`: Size of background distribution. Default: `200`.
* `distribution`: Type of background distribution: `gamma` or `normal`. Default: `gamma`.

Include parameters relevant for parallelization on a high-computing cluster, which is highly recommended for large pathway databases. For larger datasets, consider adjusting memory and time resources: 

* `processes`: Number of processes to run in parallel. Default: `0`.
* `mem`: Memory to allocate for each process (GB). Default: `10`.
* `time`: Time to allocate for each process (hours). Default: `15`.

For a full list of available parameters, run:

```
python run.py --help
```

### Basic example

```
python run.py \
    --expression my_experiment/input/expression.csv \
    --reduction my_experiment/input/reduction.csv \
    --cell_types my_experiment/input/cell_types.csv \
    --pseudotime my_experiment/input/pseudotime.csv \
    --custom_pathways my_experiment/input/gene_sets.csv \
    --output my_experiment/output
```

### Advanced example

```
python run.py \
    --expression my_experiment/input/expression.csv \
    --cell_types my_experiment/input/cell_types.csv \
    --pseudotime my_experiment/input/pseudotime.csv \
    --output my_experiment/output \
    --reduction tsne \
    --organism human \
    --pathway_database msigdb \
    --set_fraction 0.5 \
    --processes 20
```

### Visualization

For additional visualization of specific pathways, run the `plot.py` script with parameters:

* `output`: Path to the output directory containing all tool results.
* `pathway`: Pathway name to plot.
* `cell_type`: Cell-type target column to plot.
* `lineage`: Trajectory target column to plot.

For example:

```
python plot.py \
    --output my_experiment/output \
    --pathway GOBP_POSITIVE_REGULATION_OF_MONOCYTE_DIFFERENTIATION \
    --cell_type Monocyte
```


## Output

The folder specified in `--output` will include the following upon completion of the run:

* **Preprocessed data**, including `expression.csv`, `reduction.csv`,   `cell_types.csv`, `pseudotime.csv`, `gene_sets.csv`, and `targets.png`.
* A `cache` folder containing saved background scores, which can be reused for future similar runs.
* A `tmp` folder with per-batch results and reports.
* **Full results**, comprising:
    * Tables of all gene annotation results: `cell_type_classification.csv` and `pseudotime_regression.csv`.
    * Corresponding p-value matrices: `p_values_celltypes.csv` and `p_values_pseudotime.csv`.
    * Plots: `p_values_celltypes_prediction_using_all_pathways.png` and `p_values_pseudotime_prediction_using_all_pathways.png`.
* **Top results**, consisting of:
    * Lists of top gene annotations for each target: `top_celltypes_pathways.csv` and `top_pseudotime_pathways.csv`.
    * Corresponding p-value plots: `p_values_celltypes_prediction.png` and `p_values_pseudotime_prediction.png`.
    * Per-result plots in the `pathways` folder under `cell_types` and `pseudotime`.


## Results

Our [portal](https://nachmanilab.shinyapps.io/phoenix_results) provides easy access to the results of *Phoenix*, applied to datasets on embryonic development and hematopoietic stem cell differentiation across several organisms.
