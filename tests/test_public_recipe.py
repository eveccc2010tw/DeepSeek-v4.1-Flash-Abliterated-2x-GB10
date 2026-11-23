import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PublicRecipeTests(unittest.TestCase):
    def test_real_ab_records_are_comparable(self):
        cases = json.loads((ROOT / 'examples/matched-speed-cases.json').read_text())
        result = module('summarize-ab').summarize(ROOT / 'records/ab', cases)
        self.assertEqual(result['requests'], 24)
        self.assertEqual(result['generated_tokens'], {'A': 2688, 'B': 2688})
        self.assertAlmostEqual(result['scenarios']['short']['change_percent'], 0.476281842564, places=8)

    def test_ab_rejects_hash_cache_and_missing_case_changes(self):
        cases = json.loads((ROOT / 'examples/matched-speed-cases.json').read_text())
        summarize = module('summarize-ab').summarize
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / 'ab'
            shutil.copytree(ROOT / 'records/ab', records)
            path = records / 'B-1.json'
            original = path.read_text()
            for kind in ('output', 'cache', 'missing'):
                data = json.loads(original)
                if kind == 'output': data['results'][0]['output_sha256'] = '0' * 64
                elif kind == 'cache': data['results'][0]['usage']['prompt_tokens_details']['cached_tokens'] = 64
                else: data['results'].pop()
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError): summarize(records, cases)
            path.write_text(original)

    def test_ssh_adapter_quotes_remote_arguments(self):
        supervisor = module('supervise-native-pair')
        with patch.object(supervisor.subprocess, 'check_output', return_value='true\n') as call:
            self.assertEqual(supervisor.remote('spark-head', ['docker', 'inspect', 'a; echo bad']), 'true')
        argv = call.call_args.args[0]
        self.assertEqual(argv[:-1], ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '--', 'spark-head'])
        self.assertEqual(argv[-1], "docker inspect 'a; echo bad'")

    def test_example_pair_agrees_on_current_profile(self):
        prepare = module('prepare-mia-zig-container')
        specs = [json.loads((ROOT / f'examples/hostspec-{name}.json').read_text()) for name in ('head','worker')]
        for spec in specs:
            cfg = prepare.candidate_config(spec, 'sha256:test', {'tf.ref': 'test', 'tf.patches': 'public-digest'})
            env = dict(item.split('=', 1) for item in cfg['Env'])
            self.assertEqual(env['TF_DS_ROUND_ROWS'], '48')
            self.assertEqual(env['TF_DS_POOL'], '2101248')
            self.assertEqual(env['TF_DS_RECIPE_PATCHES'], 'public-digest')
            self.assertEqual(env['TF_DS_RDMA_CPUS'], '0')
            self.assertEqual(env['TF_DS_ENGRAM_LEAD'], '1')
            if spec['rank'] == 0:
                self.assertEqual(env['TF_DS_COPY_ROWS'], '48')
                self.assertEqual(cfg['Cmd'][cfg['Cmd'].index('--prompt-cache-gib')+1], '0.5')
