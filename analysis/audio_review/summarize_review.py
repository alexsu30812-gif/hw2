"""Audit the complete audio-input run and retain its failed calibration.

Completion means every WAV frame reached the critic, not that its descriptions
are valid. The critic failed the silence control; no listener rating is inferred.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/audio_review'


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    coverage = read('results/audio_review/input_coverage.json')
    raw = read('results/audio_review/raw_audio_observations.json')
    smoke = read('analysis/audio_review_smoke.json')
    cases = read('analysis/case_studies/bar_case_studies.json')
    if not raw['completed'] or raw['pieces_completed'] != 9:
        raise ValueError('All nine complete audio inputs are required')
    if raw['provenance']['input_coverage_sha256'] != sha('results/audio_review/input_coverage.json'):
        raise ValueError('Audio-input coverage changed')
    by_id = {p['blind_id']: p for p in coverage['pieces']}
    by_case = {(p['prompt_id'], p['configuration_id']): p for p in cases['basic_b']['rows']}
    rows, frames, clips = [], 0, 0
    for piece in raw['pieces']:
        source = by_id[piece['blind_id']]
        if (not piece['full_frame_coverage_verified']
                or not piece['all_frontend_waveform_hashes_match']
                or source['source_wav_sha256'] != sha(piece['source_wav'])
                or piece['source_wav_sha256'] != source['source_wav_sha256']
                or piece['segments'] != source['segments']):
            raise ValueError('Source audio or segment inventory changed')
        cursor = 0
        if len(piece['frontend_extractions']) != len(source['segments']):
            raise ValueError('An audio clip was not encoded')
        for segment, extraction in zip(source['segments'], piece['frontend_extractions']):
            if (segment['start_frame'] != cursor
                    or segment['end_frame'] - cursor != segment['frames']
                    or extraction['input_frames'] != segment['frames']
                    or extraction['feature_shape'] != [1, 128, 3000]
                    or not 0 < extraction['input_frames'] <= 480000):
                raise ValueError('Gap, overlap, truncation or frontend mismatch')
            cursor = segment['end_frame']
        if cursor != source['resampled_frames'] or not source['segments'][-1]['includes_release_tail']:
            raise ValueError('Full waveform including release was not covered')
        # Preserve malformed raw output; this recovery is only for a schema audit.
        try:
            parsed = json.loads(piece['raw_response'].replace('\\"', '"'))
        except (ValueError, TypeError):
            parsed = {}
        named = {str(p.get('audio')) for p in parsed.get('clips', []) if isinstance(p, dict) and 'audio' in p}
        missing = sorted({str(s['clip']) for s in source['segments']} - named)
        case = by_case[(piece['prompt_id'], piece['configuration_id'])]
        rows.append({
            'blind_id': piece['blind_id'], 'prompt_id': piece['prompt_id'],
            'configuration_id': piece['configuration_id'], 'source_wav': piece['source_wav'],
            'source_wav_sha256': piece['source_wav_sha256'],
            'source_seconds': source['original_seconds'], 'encoded_clips': len(source['segments']),
            'all_resampled_frames_encoded': True, 'release_tail_included': True,
            'clip_numbers_not_individually_described_in_output': missing,
            'raw_description_is_reliable_quality_evidence': False,
            'verified_bar_comparison': [{k: s[k] for k in ('bar_range', 'time_range', 'notes_per_bar', 'mean_pitch', 'p95_pitch')}
                                        for s in case['segments']],
            'interpretation': 'The complete WAV was encoded, but the failed silence control prevents treating the model prose as a validated listening judgment. Bar facts are MIDI measurements, not human impressions.'})
        clips += len(source['segments'])
        frames += cursor
    if len(rows) != 9 or clips != 39 or len(by_id) != 9:
        raise ValueError('Expected nine complete pieces and 39 ordered clips')
    evidence = [
        'results/audio_review/input_coverage.json',
        'results/audio_review/raw_audio_observations.json',
        'analysis/audio_review_model_setup.json',
        'analysis/audio_review_frontend_comparison.json',
        'analysis/audio_review_smoke.json']
    summary = {
        'completed': True, 'completion_means': 'Audio-input processing completed; critic validity did not pass.',
        'source_model_input_frames_verified': True, 'human_listening_performed': False,
        'quality_assessment_valid': False, 'calibration_passed': False,
        'method': raw['method'], 'model': raw['provenance']['model_repository'],
        'coverage': {'n_pieces': 9, 'full_frame_coverage_verified': True,
                     'n_encoded_clips': clips, 'resampled_frames_encoded': frames,
                     'source_audio_seconds': coverage['source_seconds_total'],
                     'release_tails_included': True},
        'calibration': {'silence_is_zero': smoke['silence_control']['zero_waveform'],
                        'silence_raw_response': smoke['silence_control']['raw_response'],
                        'result': 'failed: the model describes music for a zero waveform',
                        'consequence': 'No subjective preference, musical cadence, chord correctness or human-listening claim is supported.'},
        'report_summary': 'All nine complete WAVs (15 min 25 s, including release tails) reached Qwen2-Audio-7B-Instruct-4bit: 39 ordered clips, each at most 30 s, with all clip embeddings in one request per piece. Every frontend input length and waveform hash was checked. Labels, MIDI statistics and preferred versions were withheld. However, an identical-prompt silence control produced invented music. This completes audio-input coverage, not a validated listening evaluation.',
        'report_rows': [
            {'prompt_id': '001', 'observation': 't08 and t10 give incompatible major/minor descriptions of the shared prompt. t10 and t12 omit clip 5 in their prose despite encoding it. Reject these unverified claims; use the preceding bar table.'},
            {'prompt_id': '002', 'observation': 'All three receive generic smooth/fade descriptions despite different density trajectories. These words establish neither a preferred version nor a resolved cadence; the per-bar MIDI comparison remains the evidence.'},
            {'prompt_id': '003', 'observation': 'The critic gives generic register/development descriptions rather than a checkable account of the t12 late density collapse and +16.19-semitone mean-pitch shift. Its text adds no validated quality judgment.'}],
        'report_limitations': 'No human listening or listener ratings were performed. The silence hallucination and omitted clip descriptions make this critic unreliable for preference or harmony claims. A synthesized release tail is not evidence of a composed cadence. Keep raw responses for audit; use MIDI for measured bar facts and listening.html for direct comparison.',
        'rows': sorted(rows, key=lambda r: (r['prompt_id'], r['configuration_id'])),
        'evidence_files': evidence,
        'evidence_sha256': {p: sha(p) for p in evidence},
        'references': [
            'Chu et al. (2024). Qwen2-Audio Technical Report. Evaluation-only audio-language critic; not a composition model. https://arxiv.org/abs/2407.10759',
            'mlx-community/Qwen2-Audio-7B-Instruct-4bit (Apache-2.0). Revision c65570002626f41b4dc08b7b54f42f99f3e82e7f; external review weights excluded. https://huggingface.co/mlx-community/Qwen2-Audio-7B-Instruct-4bit',
            'Blaizzy/mlx-audio v0.6.1 (MIT). Review runtime; official HF Whisper frontend substituted after numerical comparison. https://github.com/Blaizzy/mlx-audio'],
    }
    target = OUT / 'audio_review_summary.json'
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'summary': str(target.relative_to(ROOT)), 'pieces': 9, 'clips': clips,
                      'seconds': coverage['source_seconds_total'], 'calibration_passed': False,
                      'omitted_clip_descriptions': {r['blind_id']: r['clip_numbers_not_individually_described_in_output'] for r in rows if r['clip_numbers_not_individually_described_in_output']}}, indent=2))


if __name__ == '__main__':
    main()
