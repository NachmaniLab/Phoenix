import os, subprocess


def to_str(value) -> str:
    if hasattr(value, 'name'):
        return value.name
    if isinstance(value, str):
        return repr(value)
    return str(value)


def get_cmd(
        func: str,
        script: str,
        args: dict[str, str],
        sbatch: bool = True,
        processes: int | None = None,
        mem: str = '1G',
        time: str = '0:30:0',
        report_path: str | None = None,
        previous_job_id: str | None = None,
        previous_processes: int | None = None,
    ) -> str:

    parsed_args = ', '.join([f'{k}={to_str(v)}' for k, v in args.items()])
    parsed_args = parsed_args.replace("'", '\\"')
    
    script = f'scripts.{script}' if not os.path.exists(f'{script}.py') else script
    python_cmd = (
        f"python -c 'from {script} import {func}; "
        f"{func}({parsed_args})' "
    )
    if not sbatch:
        return python_cmd
    
    report_info = '%j' if not processes else r'%A_%a'
    sbatch_cmd = (
        f"sbatch --job-name={func} --mem={mem} --time={time} "
        f"--output={report_path}/{report_info}_{func}.out "
        f"--error={report_path}/{report_info}_{func}.err "
        f"--wrap=\"{python_cmd}\" "
    )

    sbatch_cmd += f'--array=1-{processes} ' if processes else ''

    if previous_job_id:
        if previous_processes:
            sbatch_cmd += f"--dependency=afterok:{','.join([f'{previous_job_id}_{p + 1}' for p in range(previous_processes)])} "
        else:
            sbatch_cmd += f"--dependency=afterok:{previous_job_id} "
    
    return sbatch_cmd


def execute_sbatch_cmd(cmd, title: str, processes: int | None = None) -> str:
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try:
        job_id = process.stdout.strip().split()[-1]
        print(f'Executing {title} as job {job_id}' + (f' ({processes} processes)' if processes else '') + '...')
        return job_id
    except:
        raise RuntimeError(f'Failed executing command {cmd} due to {process.stderr}')
