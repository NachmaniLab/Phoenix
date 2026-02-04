from scripts.computation import execute_sbatch_cmd, get_cmd


# Step 1: Setup
def run_setup_cmd(args: dict, tmp: str | None = None) -> str:
    cmd = get_cmd(
        func='setup',
        script='step_1_setup',
        args=args,
        mem='5G',
        time='0:15:0',
        report_path=tmp,
    )
    return execute_sbatch_cmd(cmd, 'initial setup')


# Step 2: Pathway Scoring
def run_pathway_scoring_cmd(args: dict, processes: int, mem: int, time: int, tmp: str, setup_job_id: str) -> str:
    cmd = get_cmd(
        func='calculate_pathway_scores',
        script='step_2_pathway_scoring',
        args=args,
        processes=processes,
        mem=f'{mem}G',
        time=f'{time}:0:0',
        report_path=tmp,
        previous_job_id=setup_job_id,
    )
    return execute_sbatch_cmd(cmd, 'pathway scoring', processes)


# Step 3: Background Scoring
def run_background_scoring_cmd(args: dict, processes: int, mem: int, time: int, tmp: str, pathway_scoring_job_id: str) -> str:
    cmd = get_cmd(
        func='calculate_background_scores',
        script='step_3_background_scoring',
        args=args,
        processes=processes,
        mem=f'{mem}G',
        time=f'{time}:0:0',
        report_path=tmp,
        previous_job_id=pathway_scoring_job_id,
        previous_processes=processes,
    )
    return execute_sbatch_cmd(cmd, 'background scoring', processes)


# Step 4: Aggregation and Statistical Evaluation
def run_aggregation_cmd(args: dict, processes: int, tmp: str, pathway_scoring_job_id: str) -> str:
    cmd = get_cmd(
        func='aggregate',
        script='step_4_aggregation',
        args=args,
        report_path=tmp,
        mem='10G',
        time='5:0:0',
        previous_job_id=pathway_scoring_job_id,
        previous_processes=processes,
    )
    return execute_sbatch_cmd(cmd, 'aggregation')
