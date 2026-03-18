from scripts.args import get_plot_args
from scripts.visualization import plot, plot_experiment
from scripts.output import read_args


def plot_tool(
        cell_type: list[str],
        lineage: list[str],
        pathway: list[str],
        top: int,
        all: bool,
        output: str,
    ):
    if not pathway:
        args = read_args(output)
        plot(
            output,
            fdr_threshold=args['fdr_threshold'],
            corrected_effect_size_threshold=args['corrected_effect_size_threshold'],
            importance_lower_threshold=args['importance_lower_threshold'],
            importance_gene_fraction_threshold=args['importance_gene_fraction_threshold'],
            top=top, all=all
        )

    for p in pathway:
        for c in cell_type:
            plot_experiment(output, c, p, 'cell_types', 'cell_type_classification', 'cell_types')
        for l in lineage:
            plot_experiment(output, l, p, 'pseudotime', 'pseudotime_regression', 'pseudotime')


if __name__ == '__main__':
    args = get_plot_args()
    plot_tool(**vars(args))
