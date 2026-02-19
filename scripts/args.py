import argparse, os
from scripts.consts import *
from scripts.output import create_dir, save_args
from scripts.utils import get_full_path, parse_missing_args, str2enum, str2bool


### Run ###


def parse_run_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Input data
    parser.add_argument('--expression', type=str, required=True,
                        help='Path to single-cell raw expression data where rows represent cells and columns represent gene symbols (CSV file or 10x MTX directory containing matrix.mtx, features.tsv, and barcodes.tsv)')
    parser.add_argument('--cell_types', type=str,
                        help='Path to cell-type annotations where rows represent cells and first column presents cell-types (CSV file)')
    parser.add_argument('--pseudotime', type=str,
                        help='Path to pseudo-time values where rows represent cells and columns include names of different trajectories (CSV file)')
    parser.add_argument('--reduction', type=str, default=REDUCTION,
                        help='Path to dimensionality reduction coordinates where rows represent cells and columns include names of the first two components (CSV file), or a dimensionality reduction method: ' + ', '.join(REDUCTION_METHODS))

    # Data preprocessing
    parser.add_argument('--preprocessed', action='store_true', default=False,
                        help='Whether expression data are log-normalized')
    parser.add_argument('--exclude_cell_types', type=str, nargs='*',
                        help='Cell-type to exclude from analysis')
    parser.add_argument('--exclude_lineages', type=str, nargs='*',
                        help='Lineage to exclude from analysis')

    # Pathway annotations
    parser.add_argument('--organism', type=str,
                        help='Organism name for pathway annotations')
    parser.add_argument('--pathway_database', type=str, nargs='*',
                        help='Known pathway database: ' + ', '.join(DATABASES))
    parser.add_argument('--custom_pathways', type=str, nargs='*',
                        help='Path to custom gene sets where columns represent set names and rows include gene symbols (CSV file)')

    # Feature selection
    parser.add_argument('--feature_selection', type=str, default=FEATURE_SELECTION,
                        help='Feature selection method applied to each gene set: ' + ', '.join(FEATURE_SELECTION_METHODS))
    parser.add_argument('--set_fraction', type=float, default=SET_FRACTION,
                        help='Fraction of genes to select from each gene set')
    parser.add_argument('--min_set_size', type=int, default=MIN_SET_SIZE,
                        help='Minimum number of genes to select from each gene set')

    # Prediction model
    parser.add_argument('--classifier', type=str, default=CLASSIFIER,
                        help='Classification model: ' + ', '.join(CLASSIFIERS.keys()))
    parser.add_argument('--regressor', type=str, default=REGRESSOR,
                        help='Regression model: ' + ', '.join(REGRESSORS.keys()))
    parser.add_argument('--classification_metric', type=str, default=CLASSIFICATION_METRIC,
                        help='Classification score: ' + ', '.join(CLASSIFICATION_METRICS.keys()))
    parser.add_argument('--regression_metric', type=str, default=REGRESSION_METRIC,
                        help='Regression score: ' + ', '.join(REGRESSION_METRICS.keys()))
    parser.add_argument('--cross_validation', type=int, default=CROSS_VALIDATION,
                        help='Number of cross-validation folds')
    parser.add_argument('--background_mode', type=str, default=BackgroundMode.AUTO.name,
                        help='Background mode for p-value estimation: `real` uses real pathway scores, `random` uses scores from random gene sets, and `auto` selects automatically based on the number of gene sets')
    parser.add_argument('--repeats', type=int, default=REPEATS,
                        help='Size of background distribution')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for reproducibility')
    parser.add_argument('--distribution', type=str, default=DISTRIBUTIONS[0],
                        help='Type of background distribution: ' + ', '.join(DISTRIBUTIONS))
    parser.add_argument('--effect_size_threshold', type=float, default=EFFECT_SIZE_THRESHOLD,
                        help='Expression threshold for effect size calculation. Genes with expression below this value will be masked. Default: None (no masking)')
    parser.add_argument('--corrected_effect_size', type=str, default='True',
                        help='Whether to correct effect size by target label. Default: True')
    
    # Output
    parser.add_argument('--processes', type=int, default=0,
                        help='Number of processes to run in parallel')
    parser.add_argument('--mem', type=int, default=MEM,
                        help='Memory to allocate for each process (GB)')
    parser.add_argument('--time', type=int, default=TIME,
                        help='Time to allocate for each process (hours)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Whether to print progress messages')
    
    return parser.parse_args()


def process_run_args(args):

    args.expression = get_full_path(args.expression)
    args.cell_types = get_full_path(args.cell_types) if args.cell_types else None
    args.pseudotime = get_full_path(args.pseudotime) if args.pseudotime else None
    args.reduction = get_full_path(args.reduction) if os.path.exists(args.reduction) else args.reduction.lower().replace('-', '').replace('_', '').replace(' ', '')
    
    args.organism = args.organism.lower() if args.organism else None
    if args.pathway_database is not None:
        args.pathway_database = [db.lower() for db in args.pathway_database]
        from scripts.pathways import get_msigdb_organism
        if 'msigdb' in args.pathway_database and len(args.pathway_database) > 1 and args.organism and get_msigdb_organism(args.organism):
            args.pathway_database = ['msigdb']
            print(f'MSigDB already includes the other annotation databases for {args.organism} - automatically removing other annotations')
    else:
        args.pathway_database = []
    args.custom_pathways = args.custom_pathways if args.custom_pathways else []
    
    args.classifier = args.classifier.upper()
    args.regressor = args.regressor.upper()
    args.classification_metric = args.classification_metric.lower().replace(' ', '_')
    args.regression_metric = args.regression_metric.lower().replace(' ', '_')
    args.feature_selection = args.feature_selection.upper() if args.feature_selection else None
    args.distribution = args.distribution.lower()
    args.corrected_effect_size = str2bool(args.corrected_effect_size)
    args.background_mode = str2enum(BackgroundMode, args.background_mode)

    create_dir(args.output)
    args.output = get_full_path(args.output)

    args.cache = os.path.join(args.output, 'cache')
    create_dir(args.cache)

    args.tmp = os.path.join(args.output, 'tmp')
    if args.processes:
        create_dir(args.tmp)
    
    return args


def validate_run_args(args):
    assert args.cell_types is not None or args.pseudotime is not None, 'Provide at least `cell_types` or `pseudotime`'
    assert os.path.exists(args.reduction) or args.reduction in REDUCTION_METHODS
    assert args.organism is not None or args.pathway_database is None, 'Provide `organism` for pathway annotations'
    assert args.pathway_database is not None or args.custom_pathways is not None, 'Provide at least `pathway_database` or `custom_pathways`'
    assert args.pathway_database is None or all([db in DATABASES for db in args.pathway_database])
    assert args.classifier in CLASSIFIERS
    assert args.regressor in REGRESSORS
    assert args.classification_metric in CLASSIFICATION_METRICS.keys()
    assert args.regression_metric in REGRESSION_METRICS.keys()
    assert not args.feature_selection or args.feature_selection in FEATURE_SELECTION_METHODS
    assert 2 <= args.cross_validation <= 10
    assert args.repeats >= 5
    assert args.seed > 0
    assert args.distribution in DISTRIBUTIONS
    assert 0 < args.set_fraction <= 1
    assert args.min_set_size >= 2
    assert not args.processes or args.processes >= 0
    assert not args.processes or args.mem > 0
    assert not args.processes or args.time > 0


def get_run_args():
    args = parse_run_args()
    args = parse_missing_args(args)
    args = process_run_args(args)
    validate_run_args(args)
    save_args(args, args.output)
    return args


### Plot ###


def parse_plot_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--cell_type', type=str, nargs='*',
                        help='Cell-type target column to plot')
    parser.add_argument('--lineage', type=str, nargs='*',
                        help='Trajectory target column to plot')
    parser.add_argument('--pathway', type=str, nargs='*',
                        help='Pathway to plot')
    parser.add_argument('--top', type=int, default=None,
                        help='Number of top pathways to plot')
    parser.add_argument('--all', action='store_true', default=False,
                        help='Whether to plot all pathways for all cell-type and lineage targets')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output directory containing tool results')
    
    return parser.parse_args()


def process_plot_args(args):
    args.cell_type = args.cell_type if args.cell_type and not args.all else []
    args.lineage = args.lineage if args.lineage and not args.all else []
    args.pathway = args.pathway if args.pathway and not args.all else []
    args.output = get_full_path(args.output)
    return args


def validate_plot_args(args):
    if args.pathway:
        assert args.cell_type or args.lineage, 'Provide either `cell_type` or `lineage` target column'


def get_plot_args():
    args = parse_plot_args()
    args = parse_missing_args(args)
    args = process_plot_args(args)
    validate_plot_args(args)
    return args
