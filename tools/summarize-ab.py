#!/usr/bin/env python3
"""Validate paired synthetic A/B records and print reproducible median results."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(records, cases):
    expected = {c['prompt_sha256']: c for c in cases}
    if len(expected) != 4 or len(cases) != 4:
        raise ValueError('Expected four distinct prompt fixtures')
    for c in cases:
        if hashlib.sha256(json.dumps(c['messages']).encode()).hexdigest() != c['prompt_sha256']:
            raise ValueError('Prompt fixture hash mismatch')
    groups = {}
    for label in ('A', 'B'):
        runs = []
        for repeat in range(1, 4):
            report = json.loads((Path(records) / f'{label}-{repeat}.json').read_text())
            seen = set()
            for r in report['results']:
                digest = r['prompt_sha256']
                if digest not in expected or digest in seen:
                    raise ValueError('Missing, unknown or duplicate prompt')
                seen.add(digest)
                c, u = expected[digest], r['usage']
                if not (r['matched'] and r['http_status'] == 200 and r['done']
                        and r['finish_reason'] == 'length'
                        and r['scenario'] == c['scenario'] and r['trial'] == c['trial']
                        and u['prompt_tokens'] == c['expected_prompt_tokens']
                        and u['completion_tokens'] == c['max_tokens']
                        and u['prompt_tokens_details']['cached_tokens'] == 0
                        and u['completion_tokens_details']['reasoning_tokens'] == 0
                        and r['reasoning_chars'] == 0):
                    raise ValueError('Request is not a matched uncached decode')
                duration = r['last_token_seconds'] - r['ttft_seconds']
                rate = r['decode_tokens_per_second']
                if duration <= 0 or rate <= 0 or abs(rate - (c['max_tokens'] - 1) / duration) > 1e-6:
                    raise ValueError('Decode timing/rate mismatch')
                runs.append(r)
            if seen != set(expected):
                raise ValueError('Incomplete round')
        groups[label] = runs
    for digest in expected:
        hashes = {r['output_sha256'] for runs in groups.values() for r in runs if r['prompt_sha256'] == digest}
        if len(hashes) != 1:
            raise ValueError('A/B output hashes differ')
    result = {'requests': sum(map(len, groups.values())),
              'generated_tokens': {k: sum(r['usage']['completion_tokens'] for r in v) for k,v in groups.items()},
              'scenarios': {}}
    for scenario in sorted({c['scenario'] for c in cases}):
        medians = {label: statistics.median(r['decode_tokens_per_second'] for r in runs if r['scenario'] == scenario)
                   for label,runs in groups.items()}
        result['scenarios'][scenario] = dict(medians, change_percent=(medians['B'] / medians['A'] - 1) * 100)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=Path(__file__).parents[1] / 'examples/matched-speed-cases.json')
    a = p.parse_args()
    print(json.dumps(summarize(a.records, json.loads(a.cases.read_text())), indent=2))


if __name__ == '__main__':
    main()
