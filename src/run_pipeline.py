"""Serial GPU training stages followed by actual, reproducible music experiments.

This driver resumes completed stages from their saved evidence. It does not
invent missing results, and stops on a failed subprocess or invalid generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RESULTS = ROOT / 'results'


def write_status(stage, **fields):
    record = {'stage': stage, 'updated_unix': time.time(), **fields}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'pipeline_status.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


def run(name, arguments):
    logs = RESULTS / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    write_status(name, state='running', command=arguments)
    with (logs / f'{name}.log').open('w') as out:
        process = subprocess.run([PYTHON] + arguments, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if process.returncode:
        write_status(name, state='failed', log=f'results/logs/{name}.log', returncode=process.returncode)
        raise RuntimeError(f'{name} failed; inspect results/logs/{name}.log')
    write_status(name, state='completed')


def train_stage(name, parent, corpus, context, batch_size, budget, lr, all_songs=False, mode='basic'):
    directory = ROOT / 'artifacts/models' / name
    history = directory / 'history.json'
    if history.exists():
        record = json.loads(history.read_text())['metadata']
        if record['processed_tokens'] >= budget and (directory / 'last.pt').exists():
            from model import load_checkpoint
            _, saved = load_checkpoint(directory / 'last.pt')
            expected = {'context': context, 'batch_size': batch_size, 'token_budget': budget,
                        'lr': lr, 'all_songs': all_songs, 'mode': mode}
            if (any(saved.get('arguments', {}).get(key) != value for key, value in expected.items())
                    or saved.get('origin', {}).get('sha256') != file_sha256(parent)
                    or saved.get('corpus_sha256') != file_sha256(ROOT / corpus)
                    or saved.get('step') != record.get('step')
                    or saved.get('processed_tokens') != record.get('processed_tokens')
                    or set(saved.get('seen_song_ids', [])) != set(saved.get('training_song_ids', []))):
                raise ValueError(f'Completed stage {name} conflicts with the requested provenance; preserve it and use a new output directory')
            return directory
    arguments = ['src/train.py', '--corpus', corpus, '--initialize-from', str(parent),
                 '--output-dir', str(directory), '--mode', mode, '--steps', '16000',
                 '--context', str(context), '--batch-size', str(batch_size),
                 '--token-budget', str(budget), '--lr', str(lr), '--warmup', '100',
                 '--augment', '--device', 'mps', '--eval-every', '500', '--log-every', '50',
                 '--keep-steps', '500', '1000']
    if all_songs:
        arguments.append('--all-songs')
    run(name, arguments)
    return directory


CONFIGURATIONS = [
    {'id': 't08', 'temperature': .8, 'top_p': .95, 'top_k': 0},
    {'id': 't10', 'temperature': 1., 'top_p': .95, 'top_k': 0},
    {'id': 't12', 'temperature': 1.2, 'top_p': .95, 'top_k': 0},
    {'id': 'k16', 'temperature': 1., 'top_p': 1., 'top_k': 16},
    {'id': 'k32', 'temperature': 1., 'top_p': 1., 'top_k': 32},
]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_generation(manifest, checkpoint, task, count, configuration, bars, conditioning=None):
    """Resume only outputs from the same model/settings whose MIDI hashes still match."""
    try:
        result = json.loads(Path(manifest).read_text())
        if (not result.get('complete') or result.get('task') != task
                or result.get('checkpoint_sha256') != file_sha256(checkpoint)
                or len(result.get('samples', [])) != count
                or any(result.get('configuration', {}).get(k) != v for k, v in configuration.items())
                or (conditioning is not None and result.get('conditioning') != conditioning)):
            return False
        identities = set()
        for sample in result['samples']:
            path = Path(sample['midi_path'])
            if not path.is_absolute():
                path = ROOT / path
            if (sample.get('bars') != bars or file_sha256(path) != sample.get('midi_sha256')
                    or sample['song_name'] in identities):
                return False
            identities.add(sample['song_name'])
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def generate_basic(device='cpu'):
    for config in CONFIGURATIONS:
        name = config['id']
        for task, folder, count in [('unconditional', 'basic_a', 20), ('continuation', 'basic_b', 3)]:
            if task == 'continuation' and name not in ('t08', 't10', 't12'):
                continue
            output = ROOT / 'results' / folder / name
            manifest = output / 'manifest.json'
            expected = {key: config[key] for key in ('temperature', 'top_p', 'top_k')}
            expected.update(seed=20261022, device=device)
            if valid_generation(manifest, ROOT / 'artifacts/models/basic_final.pt',
                                task, count, expected, 32):
                continue
            run(f'{folder}_{name}', ['src/infer.py', '--checkpoint', 'artifacts/models/basic_final.pt',
                '--task', task, '--output-dir', str(output), '--num-samples', str(count),
                '--temperature', str(config['temperature']), '--top-p', str(config['top_p']),
                '--top-k', str(config['top_k']), '--seed', '20261022', '--device', device])


def choose_generation_device():
    """Compare exclusive-device cached decoding after all training is finished."""
    import torch
    from model import load_checkpoint
    model, _ = load_checkpoint(ROOT / 'artifacts/models/basic_final.pt')
    model.eval(); torch.set_num_threads(1)
    torch.manual_seed(9)
    ids = torch.randint(0, model.config.vocab_size, (1, 256))
    token = ids[:, :1]
    records, reference = [], None
    with torch.inference_mode():
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            model.to(device)
            _, _, cache = model(ids.to(device), use_cache=True)
            times, logits = [], []
            for _ in range(25):
                if device == 'mps': torch.mps.synchronize()
                started = time.perf_counter()
                scores, _, cache = model(token.to(device), past=cache, use_cache=True)
                if device == 'mps': torch.mps.synchronize()
                times.append(time.perf_counter() - started)
                logits.append(scores.float().cpu())
            joined = torch.cat(logits)
            if reference is None: reference = joined
            difference = float((joined - reference).abs().max())
            records.append({'device': device, 'cached_step_seconds': sum(times[5:]) / 20,
                            'max_absolute_logit_difference_vs_cpu': difference})
            del cache
    selected = min((row for row in records if row['max_absolute_logit_difference_vs_cpu'] < 1e-3),
                   key=lambda row: row['cached_step_seconds'])['device']
    (RESULTS / 'final_generation_device_probe.json').write_text(json.dumps({
        'after_training_exclusive_device_probe': True, 'selected': selected, 'measurements': records,
        'limitation': 'Small latency and numerical comparison only; sampling across devices can diverge despite close logits.'}, indent=2) + '\n')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-basic-only', action='store_true')
    parser.add_argument('--wait-for-refined', action='store_true')
    parser.add_argument('--generation-device', choices=['cpu', 'mps'], default='cpu')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.generate_basic_only:
        generate_basic(args.generation_device)
        return
    plan = {'generation_configurations': CONFIGURATIONS,
            'piece_seed_schedule': '20261022 + piece_index, reused for every configuration',
            'context_ablation': {'parent': 'basic_refined/best.pt', 'contexts': [2048, 4096],
                                  'new_nonpadding_target_budget_each': 4000000,
                                  'common_diagnostic_validation_context': 2048,
                                  'rule': 'Compare the last completed equal-budget checkpoint from each branch on deterministic held-out CE before all-song refit'},
            'basic_a_selection_rule': 'Closest H4/GS mean vector to real reference after scaling each by real-reference per-piece standard deviation; not a course grading formula. Report density, repetition and validity separately. No claim of human listening.',
            'all_song_refit': 'Final models include all896 supplied training songs; earlier45-song reference is not held-out from final models.',
            'bonus_diagnostic_parent': 'basic_refined/best.pt, never an all896-trained parent',
            'failure_policy': 'Abort and preserve failed logs; no fabricated or template fallback music'}
    plan_path = RESULTS / 'experiment_plan.json'
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    refined = ROOT / 'artifacts/models/basic_refined'
    if args.wait_for_refined:
        started = time.monotonic()
        while True:
            history = refined / 'history.json'
            if history.exists():
                try:
                    metadata = json.loads(history.read_text())['metadata']
                    if metadata.get('step', 0) >= 4000:
                        break
                except json.JSONDecodeError:
                    pass
            if time.monotonic() - started > 7200:
                raise RuntimeError('Timed out waiting for diagnostic refinement')
            write_status('waiting_for_basic_refined', state='waiting')
            time.sleep(30)
    if not (refined / 'best.pt').exists():
        raise FileNotFoundError('Run basic_refined training first')
    parent = refined / 'best.pt'
    branches = []
    for context, batch in [(2048, 4), (4096, 2)]:
        directory = train_stage(f'context_{context}', parent, 'artifacts/tokens_basic.npz',
                                context, batch, 4000000, 0.00008)
        from model import load_checkpoint
        _, metadata = load_checkpoint(directory / 'last.pt')
        branches.append({'context': context, 'checkpoint': str(directory / 'last.pt'),
                         'validation_loss': metadata['validation_loss'],
                         'processed_tokens': metadata['processed_tokens'], 'parent_sha256': metadata['origin']['sha256']})
    best = min(branches, key=lambda item: item['validation_loss'])
    (RESULTS / 'context_comparison.json').write_text(json.dumps({'branches': branches, 'selected': best,
        'comparison_budget_note': 'Compare each last checkpoint at4m nonpadding targets, with at most one optimizer-step overshoot. Intermediate best checkpoints are not used for the equal-budget comparison.',
        'not_course_grade': True}, indent=2) + '\n')
    final_basic = train_stage('basic_all', Path(best['checkpoint']), 'artifacts/tokens_basic.npz',
                               best['context'], 4 if best['context'] == 2048 else 2,
                               28000000, 0.00005, all_songs=True)
    shutil.copy2(final_basic / 'last.pt', ROOT / 'artifacts/models/basic_final.pt')
    roles = train_stage('roles_diagnostic', parent, 'artifacts/tokens_roles.npz', 2048, 4,
                         8000000, 0.00015, mode='roles')
    roles_final = train_stage('roles_all', roles / 'best.pt', 'artifacts/tokens_roles.npz', 2048, 4,
                              12000000, 0.00005, all_songs=True, mode='roles')
    shutil.copy2(roles_final / 'last.pt', ROOT / 'artifacts/models/roles_final.pt')
    generation_device = choose_generation_device()
    generate_basic(generation_device)
    bonus_manifest = RESULTS / 'bonus/manifest.json'
    if not valid_generation(bonus_manifest, ROOT / 'artifacts/models/roles_final.pt',
                            'accompaniment', 10, {'temperature': 1., 'top_p': .95,
                            'top_k': 0, 'seed': 20261022, 'device': generation_device,
                            'bars': 16, 'conditioning': 'normal'}, 16, 'normal'):
        run('bonus_generation', ['src/generate_bonus.py', '--checkpoint', 'artifacts/models/roles_final.pt',
                                 '--output-dir', 'results/bonus', '--temperature', '1.', '--top-p', '.95',
                                 '--seed', '20261022', '--device', generation_device])
    ablation_manifest = RESULTS / 'bonus_ablation_no_melody/manifest.json'
    if not valid_generation(ablation_manifest, ROOT / 'artifacts/models/roles_final.pt',
                            'accompaniment', 10, {'temperature': 1., 'top_p': .95,
                            'top_k': 0, 'seed': 20261022, 'device': generation_device,
                            'bars': 16, 'conditioning': 'empty-melody'}, 16, 'empty-melody'):
        run('bonus_empty_melody_ablation', ['src/generate_bonus.py', '--checkpoint',
            'artifacts/models/roles_final.pt', '--output-dir', 'results/bonus_ablation_no_melody',
            '--temperature', '1.', '--top-p', '.95', '--seed', '20261022',
            '--device', generation_device, '--conditioning', 'empty-melody'])
    write_status('training_and_generation_complete', state='completed',
                 next='Evaluate all saved samples, render listening samples, compile report and package, publish verified files')


if __name__ == '__main__':
    main()
