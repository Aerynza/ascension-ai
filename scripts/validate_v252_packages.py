"""Validate curriculum integrity. This does not evaluate a trained model."""
import json
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
GROUPS = {'reasoning', 'execution', 'recovery', 'memory', 'verification', 'independence'}

def normalize(text):
    return ' '.join(text.casefold().split())

def validate(train, benchmark, prior):
    if len(train) != 24 or len(benchmark['cases']) != 12:
        raise ValueError('expected 24 training examples and 12 benchmark specifications')
    seen, prompts = set(), set()
    groups = Counter()
    prior_prompts = {normalize(r['user']) for r in prior}
    prior_ids = {r['id'] for r in prior}
    for row in train:
        rid = row['id']
        if not isinstance(rid, str) or not rid or rid in seen or rid in prior_ids:
            raise ValueError('duplicate or missing id')
        seen.add(rid)
        if row['shell'] not in {'ap', 'nexus_home', 'nexus_family', 'sprout'}:
            raise ValueError('invalid canonical training shell')
        if row.get('provenance') != 'ai_authored_synthetic' or row.get('source_kind') != 'synthetic':
            raise ValueError('synthetic provenance required')
        if row.get('review_status') != 'pending_human_review':
            raise ValueError('review status cannot imply unperformed review')
        for field in ('user','assistant'):
            if not isinstance(row[field], str) or len(row[field].strip()) < 20:
                raise ValueError('missing worked dialogue')
        prompt = normalize(row['user'])
        if prompt in prompts or prompt in prior_prompts:
            raise ValueError('prompt overlap')
        prompts.add(prompt)
        groups[row['package'].removeprefix('v252_')] += 1
    if set(groups) != GROUPS or set(groups.values()) != {4}:
        raise ValueError('unbalanced curriculum')
    families = Counter()
    for case in benchmark['cases']:
        if case['id'] in seen:
            raise ValueError('duplicate benchmark id')
        seen.add(case['id'])
        prompt = normalize(case['task'])
        if prompt in prompts or prompt in prior_prompts:
            raise ValueError('benchmark prompt overlap')
        prompts.add(prompt)
        if case.get('result') != 'not_run' or case.get('fixture_status') != 'requires_independent_fixture':
            raise ValueError('specification must not claim execution')
        if len(case['acceptance']) < 2 or not case['critical_failures']:
            raise ValueError('acceptance and critical failure rubrics required')
        families[case['family']] += 1
    if set(families) != GROUPS or set(families.values()) != {2}:
        raise ValueError('unbalanced benchmarks')
    return {'training_records':24, 'benchmark_specs':12, 'groups':dict(groups),
            'gpu_trained':False, 'model_evaluated':False, 'frontier_parity_established':False}

def main():
    train = [json.loads(s) for s in (ROOT/'evals/training/aerynza_v252_execution_train.jsonl').read_text().splitlines() if s.strip()]
    benchmark = json.loads((ROOT/'evals/training/aerynza_v252_execution_benchmark.json').read_text())
    # Explicit list: do not glob benchmarks into a training curriculum.
    prior=[]
    for name in ('aerynza_v251_foundation_train.jsonl','aerynza_v251_foundation_heldout.jsonl'):
        prior.extend(json.loads(s) for s in (ROOT/'evals/training'/name).read_text().splitlines() if s.strip())
    print(json.dumps(validate(train,benchmark,prior),indent=2))

if __name__ == '__main__':
    main()
