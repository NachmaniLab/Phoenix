import time as runtime
import numpy as np
import pandas as pd
from scripts.consts import TARGET_COL, BackgroundMode
from scripts.output import aggregate_batch_results, load_background_scores, save_csv
from scripts.prediction import compare_scores
from scripts.utils import correct_effect_size, define_background, adjust_p_value, format_runtime, load_total_runtime, save_step_runtime, str2enum
from scripts.visualization import plot


def calculate_p_value(
        pathway_score: float,
        distribution: str,
        set_size: int,
        background_mode: BackgroundMode,
        cache: str,
        mem_cache: dict[str, tuple[list[float], float]],
        cell_type: str | None = None,
        lineage: str | None = None,
        repeats: int | None = None,
    ) -> tuple[float, float]:
    background = define_background(set_size, background_mode, cell_type, lineage, repeats)
    if background in mem_cache:
        background_scores, background_score_mean = mem_cache[background]
    else:
        background_scores = load_background_scores(background, cache)
        background_score_mean = float(np.mean(background_scores))
        mem_cache[background] = background_scores, background_score_mean
    p_value = compare_scores(pathway_score, background_scores, distribution)
    return p_value, background_score_mean


def evaluate_and_correct_result(
        result: pd.DataFrame | None,
        result_type: str,
        background_mode: BackgroundMode,
        distribution: str,
        output: str,
        tmp: str,
        cache: str,
        repeats: int,
        corrected_effect_size: bool
    ) -> pd.DataFrame | None:
    result = result if result is not None else aggregate_batch_results(tmp, result_type)
    if result is None:
        return None
        
    is_classification = result_type == 'cell_type_classification'
    
    mem_cache: dict[str, tuple[list[float], float]] = {}
    p_values: list[float] = []
    background_score_mean_list: list[float] = []

    for row in result.itertuples(index=False):
        p_value, background_score_mean = calculate_p_value(
            pathway_score=row.pathway_score,
            set_size=row.set_size,
            cell_type=getattr(row, TARGET_COL) if is_classification else None,
            lineage=getattr(row, TARGET_COL) if not is_classification else None,
            distribution=distribution,
            background_mode=background_mode,
            cache=cache,
            repeats=repeats if background_mode == BackgroundMode.RANDOM else None,
            mem_cache=mem_cache,
        )
        p_values.append(p_value)
        background_score_mean_list.append(background_score_mean)

    result['p_value'] = p_values
    result['background_score_mean'] = background_score_mean_list
    result['fdr'] = adjust_p_value(result['p_value'])
    if corrected_effect_size:
        result['corrected_effect_size'] = correct_effect_size(result['effect_size'], result[TARGET_COL])

    save_csv(result, result_type, output, keep_index=False)
    return result


def aggregate(
        output: str,
        tmp: str,
        cache: str,
        background_mode: BackgroundMode | str,
        distribution: str,
        repeats: int,
        corrected_effect_size: bool,
        classification: pd.DataFrame | None = None,
        regression: pd.DataFrame | None = None,
        start_time: float | None = None,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    step_start = runtime.time()
    if verbose:
        print('Aggregating and evaluating results...')
    background_mode = str2enum(BackgroundMode, background_mode)

    classification = evaluate_and_correct_result(
        classification,
        result_type='cell_type_classification',
        background_mode=background_mode,  # type: ignore[arg-type]
        distribution=distribution,
        output=output,
        tmp=tmp,
        cache=cache,
        repeats=repeats,
        corrected_effect_size=corrected_effect_size,
    )
    regression = evaluate_and_correct_result(
        regression,
        result_type='pseudotime_regression',
        background_mode=background_mode,  # type: ignore[arg-type]
        distribution=distribution,
        output=output,
        tmp=tmp,
        cache=cache,
        repeats=repeats,
        corrected_effect_size=corrected_effect_size,
    )

    if verbose:
        print('Plotting results...')
    plot(output)

    step_4_time = runtime.time() - step_start
    save_step_runtime(tmp, 'step4', step_4_time)

    if verbose and start_time is not None:
        wall_clock_time = runtime.time() - start_time
        print(f"Wall-clock time: {format_runtime(wall_clock_time)}")

    if verbose:
        step_1_time = load_total_runtime(tmp, 'step1')
        step_2_time = load_total_runtime(tmp, 'step2')
        step_3_time = load_total_runtime(tmp, 'step3')

        print(f"Total time: {format_runtime(step_1_time + step_2_time + step_3_time + step_4_time)}")
        print(f"Total time setup step: {format_runtime(step_1_time)}")
        print(f"Total time pathway scoring step: {format_runtime(step_2_time)}")
        print(f"Total time background scoring step: {format_runtime(step_3_time)}")
        print(f"Total time aggregation step: {format_runtime(step_4_time)}")

    return classification, regression
