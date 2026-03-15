import unittest
from tests.interface import Test
from scripts.computation import get_cmd


class CmdTest(Test):

    def test_non_sbatch_command(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2, 'arg3': ['value3', 'value4']},
            script='my_script',
            sbatch=False
        )
        expected_cmd = 'python -c \'from scripts.my_script import my_function; my_function(arg1=\\"value1\\", arg2=2, arg3=[\\"value3\\", \\"value4\\"])\' '
        self.assertEqual(cmd, expected_cmd)
    
    def test_sbatch_command_without_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            report_path='/path/to/report'
        )
        expected_cmd = (
            'sbatch --job-name=my_function --mem=1G --time=0:30:0 --cpus-per-task=1 '
            '--output=/path/to/report/%j_my_function.out '
            '--error=/path/to/report/%j_my_function.err '
            '--wrap="python -c \'from scripts.my_script import my_function; my_function(arg1=\\"value1\\", arg2=2)\' " '
        )
        self.assertEqual(cmd, expected_cmd)

    def test_sbatch_command_with_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            processes=5,
            cpus=2,
            report_path='/path/to/report'
        )
        expected_cmd = (
            'sbatch --job-name=my_function --mem=1G --time=0:30:0 --cpus-per-task=2 '
            '--output=/path/to/report/%A_%a_my_function.out '
            '--error=/path/to/report/%A_%a_my_function.err '
            '--wrap="python -c \'from scripts.my_script import my_function; my_function(arg1=\\"value1\\", arg2=2)\' " '
            '--array=1-5 '
        )
        self.assertEqual(cmd, expected_cmd)

    def test_sbatch_command_with_dependency(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            report_path='/path/to/report',
            previous_job_id='12345'
        )
        expected_cmd = (
            'sbatch --job-name=my_function --mem=1G --time=0:30:0 --cpus-per-task=1 '
            '--output=/path/to/report/%j_my_function.out '
            '--error=/path/to/report/%j_my_function.err '
            '--wrap="python -c \'from scripts.my_script import my_function; my_function(arg1=\\"value1\\", arg2=2)\' " '
            '--dependency=afterok:12345 '
        )
        self.assertEqual(cmd, expected_cmd)
    
    def test_sbatch_command_with_dependency_and_processes(self):
        cmd = get_cmd(
            func='my_function',
            args={'arg1': 'value1', 'arg2': 2},
            script='my_script',
            sbatch=True,
            processes=5,
            report_path='/path/to/report',
            previous_job_id='12345',
            previous_processes=3
        )
        expected_cmd = (
            'sbatch --job-name=my_function --mem=1G --time=0:30:0 --cpus-per-task=1 '
            '--output=/path/to/report/%A_%a_my_function.out '
            '--error=/path/to/report/%A_%a_my_function.err '
            '--wrap="python -c \'from scripts.my_script import my_function; my_function(arg1=\\"value1\\", arg2=2)\' " '
            '--array=1-5 '
            '--dependency=afterok:12345_1,12345_2,12345_3 '
        )
        self.assertEqual(cmd, expected_cmd)


if __name__ == '__main__':
    unittest.main()
