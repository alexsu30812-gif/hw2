"""Strict wrappers around the untouched assignment metrics.

Our model uses 8 positions per annotated beat. For the official metric adapter,
Pos_q becomes Pos_(2*q), restoring the official 16-subdivision grid while retaining
128 consecutive position IDs. Real and generated pieces use the identical adapter.
Official H1/H4/GS functions and all bonus functions are loaded unchanged. An
additional GS_active diagnostic uses the native 8-subdivision grid and divides
each pair's XOR by max(meter_a,meter_b)*8, exposing both unused odd positions
and meter padding in the official 128-position normalization.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pretty_midi
import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor'))


def _official_module(name):
    spec = importlib.util.spec_from_file_location('hw2_official_'+name, ROOT/'HW2_scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BONUS = _official_module('eval_bonus')
CANONICAL_VOCAB = {'Bar': 0, **{f'Pos_{i}': i+1 for i in range(128)},
                   **{f'Pitch_{i}': i+129 for i in range(128)}}
POS_IDS, PITCH_IDS = list(range(1,129)), list(range(129,257))


def parse_bars(tokens: list[str]) -> list[dict]:
    """Validate the custom note grammar before official code can drop any tokens."""
    bars, i, track = [], 0, 'MERGED'
    tokens = list(tokens)
    while i < len(tokens):
        token = tokens[i]
        if not isinstance(token, str):
            raise ValueError('Evaluation expects token strings, not integer model IDs')
        if token in ('<BOS>','<EOS>'):
            if token == '<BOS>' and i != 0 or token == '<EOS>' and i != len(tokens)-1:
                raise ValueError('BOS/EOS must occur only at sequence boundaries')
            i += 1; continue
        if token == 'Bar':
            bars.append(dict(meter=None, tempo=None, notes=[])); track='MERGED'; i += 1; continue
        if not bars:
            raise ValueError(f'Event before first Bar: {token}')
        if token.startswith('Meter_'):
            meter=int(token.split('_')[1])
            if not 1 <= meter <= 16 or bars[-1]['meter'] is not None:
                raise ValueError('Invalid or repeated Meter event')
            bars[-1]['meter']=meter; i+=1; continue
        if token.startswith('Tempo_'):
            tempo=int(token.split('_')[1])
            if not 8 <= tempo <= 240 or bars[-1]['tempo'] is not None:
                raise ValueError('Invalid or repeated Tempo event')
            bars[-1]['tempo']=tempo; i+=1; continue
        if token in ('Track_Melody','Track_Piano'):
            track=token.split('_')[1].upper(); i+=1; continue
        if token.startswith('Pos_'):
            quartet=tokens[i:i+4]
            if len(quartet)!=4 or not all(t.startswith(p) for t,p in zip(quartet,('Pos_','Pitch_','Vel_','Dur_'))):
                raise ValueError(f'Incomplete note event near token {i}: {quartet}')
            pos,pitch,vel,dur=[int(t.split('_')[1]) for t in quartet]
            meter=bars[-1]['meter']
            if meter is None or not 0 <= pos < meter*8 or not 0<=pitch<128 or not 0<=vel<32 or not 1<=dur<=256:
                raise ValueError(f'Out-of-range note at token {i}')
            bars[-1]['notes'].append(dict(pos=pos,pitch=pitch,velocity=vel*4+2,duration=dur,track=track))
            i+=4; continue
        raise ValueError(f'Unknown/misordered token: {token}')
    if not bars or any(b['meter'] is None or b['tempo'] is None for b in bars):
        raise ValueError('Every bar must contain one Meter and one Tempo header')
    return bars


def canonical_sequence(bars):
    strings=[]
    for bar in bars:
        if bar['meter']>8:
            raise ValueError('Official 128-position/16-subdivision adapter supports at most 8 beats/bar; select a documented eligible real-data crop')
        strings.append('Bar')
        # The official functions treat a terminal bare Bar as an extra delimiter
        # and remove it. A position without a following pitch has no onset, but
        # retains an actual final empty bar in both H4 and GS.
        if not bar['notes']:
            strings.append('Pos_0')
        for n in sorted(bar['notes'],key=lambda n:(n['pos'],n['pitch'],n['duration'])):
            strings.extend((f'Pos_{2*n["pos"]}',f'Pitch_{n["pitch"]}'))
    return strings, [CANONICAL_VOCAB[t] for t in strings]


def _pattern(bar):
    return tuple(sorted((n['pos'],n['pitch'],n['duration']) for n in bar['notes']))


def piece_diagnostics(bars):
    patterns=[_pattern(b) for b in bars]
    notes=[n for b in bars for n in b['notes']]
    if not notes:
        raise ValueError('Completely empty piece')
    longest=run=1
    for a,b in zip(patterns,patterns[1:]):
        run=run+1 if a==b else 1;longest=max(longest,run)
    pairs=[]
    for a,b in itertools.combinations(bars,2):
        length=max(a['meter'],b['meter'])*8
        av={n['pos'] for n in a['notes']};bv={n['pos'] for n in b['notes']}
        pairs.append(1-len(av.symmetric_difference(bv))/length)
    count=len(bars)
    valid_h4=sum(any(b['notes'] for b in bars[i:i+4]) for i in range(count-3))
    return dict(n_bars=count,n_notes=len(notes),notes_per_bar=len(notes)/count,
                empty_bar_fraction=sum(not b['notes'] for b in bars)/count,
                unique_bar_fraction=len(set(patterns))/count,longest_identical_bar_run=longest,
                pitch_min=min(n['pitch'] for n in notes),pitch_max=max(n['pitch'] for n in notes),
                pitch_std=float(np.std([n['pitch'] for n in notes])),
                velocity_mean=float(np.mean([n['velocity'] for n in notes])),
                velocity_std=float(np.std([n['velocity'] for n in notes])),
                mean_duration_beats=float(np.mean([n['duration']/8 for n in notes])),
                GS_active=float(np.mean(pairs)),H4_valid_windows=valid_h4,
                H4_total_windows=max(0,count-3),H4_skipped_empty_windows=max(0,count-3)-valid_h4)


def metric_summary(rows, seed=8173):
    rng=np.random.default_rng(seed)
    frame=pd.DataFrame(rows)
    summary={}
    for name in frame.select_dtypes(include='number').columns:
        values=frame[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite metric {name}')
        boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
        summary[name]=dict(mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.,
                           median=float(np.median(values)),min=float(values.min()),max=float(values.max()),
                           mean_ci95=np.quantile(boot,[.025,.975]).tolist())
    return summary


def evaluate_basic_records(records, output_dir, expected_count=20, expected_bars=32, tag='basic'):
    """records: [{song_name: unique str, tokens: list[str]}]; never silently skip."""
    basic = _official_module('eval_basic')
    records=list(records);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    if len(records)!=expected_count:
        raise ValueError(f'Expected {expected_count} pieces, found {len(records)}')
    names=[r['song_name'] for r in records]
    if len(set(names))!=len(names):
        raise ValueError('Duplicate song names')
    rows=[];canonical_records=[]
    for record in records:
        bars=parse_bars(record['tokens'])
        if len(bars)!=expected_bars:
            raise ValueError(f'{record["song_name"]}: expected {expected_bars} bars, got {len(bars)}')
        strings,seq=canonical_sequence(bars)
        if seq.count(CANONICAL_VOCAB['Bar'])!=expected_bars or seq[-1]==CANONICAL_VOCAB['Bar']:
            raise ValueError('Canonical adapter lost an evaluated bar')
        row=dict(song_name=record['song_name'],
                 H1=float(basic.compute_piece_pitch_entropy(seq,1,0,PITCH_IDS)),
                 H4=float(basic.compute_piece_pitch_entropy(seq,4,0,PITCH_IDS)),
                 GS=float(basic.compute_piece_groove_similarity(seq,0,POS_IDS,PITCH_IDS)),
                 **piece_diagnostics(bars))
        if not all(np.isfinite(v) for v in row.values() if isinstance(v,(int,float))):
            raise ValueError(f'Nonfinite piece metric: {record["song_name"]}')
        rows.append(row);canonical_records.append(dict(song_name=record['song_name'],tokens=strings))
    pd.DataFrame(rows).to_csv(output_dir/f'{tag}_per_piece.csv',index=False)
    (output_dir/f'{tag}_canonical_sequences.json').write_text(json.dumps(canonical_records)+'\n')
    (output_dir/'canonical_tok2idx.json').write_text(json.dumps(CANONICAL_VOCAB,indent=2)+'\n')
    result=dict(tag=tag,expected_pieces=expected_count,evaluated_pieces=len(rows),skipped_pieces=0,
                expected_bars=expected_bars,adapter='model Pos8 -> official Pos16 by x2; 128 positions; identical for real/generated; empty bars carry a position-only event (no pitch or onset) so the official trailing-Bar rule retains them',
                uncertainty='2000 whole-piece bootstrap draws, fixed seed8173; not overlapping H4 windows',
                summary=metric_summary(rows),source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [ROOT/'HW2_scripts/eval_basic.py',ROOT/'vendor/MusDr/musdr/side_utils.py',Path(__file__)]})
    (output_dir/f'{tag}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def _note_tuples(notes,limit=None):
    return sorted((n.pitch,n.velocity,float(n.start),float(n.end)) for n in notes if limit is None or n.start<limit-1e-6)


def check_note_identity(reference, generated, tolerance=.003):
    a,b=_note_tuples(reference),_note_tuples(generated)
    if len(a)!=len(b):
        raise ValueError(f'Preserved note count differs: {len(a)} versus {len(b)}')
    max_delta=0.
    for aa,bb in zip(a,b):
        if aa[:2]!=bb[:2]:raise ValueError('Preserved pitch or velocity differs')
        delta=max(abs(aa[2]-bb[2]),abs(aa[3]-bb[3]));max_delta=max(max_delta,delta)
        if delta>tolerance:raise ValueError(f'Preserved note timing differs by {delta:.6f}s')
    return dict(notes=len(a),max_timing_difference_seconds=max_delta,tolerance_seconds=tolerance)


def validate_continuation(prompt_midi,generated_midi,beat_path,tolerance=.003):
    ref=pretty_midi.PrettyMIDI(str(prompt_midi));gen=pretty_midi.PrettyMIDI(str(generated_midi))
    times,down=BONUS.load_gt_beats(str(beat_path));bars=BONUS.GTBars(times,down)
    if bars.n_bars!=8:raise ValueError(f'Prompt contains {bars.n_bars} rather than 8 bars')
    boundary=bars.bar_end(7)
    ref_notes=[n for inst in ref.instruments for n in inst.notes if n.start<boundary-1e-6]
    gen_notes=[n for inst in gen.instruments for n in inst.notes if n.start<boundary-1e-6]
    identity=check_note_identity(ref_notes,gen_notes,tolerance)
    continuation=[n for inst in gen.instruments for n in inst.notes if n.start>=boundary-1e-6]
    if not continuation:raise ValueError('Continuation has no generated notes')
    identity.update(boundary_seconds=float(boundary),continuation_notes=len(continuation))
    return identity


def validate_annotation_clock(midi_path, beat_path, expected_bars=None, tolerance=.004):
    """Check exported annotation quarters/meters and retained final silent duration."""
    from music_codec import BeatMap
    grid=BeatMap.from_file(beat_path)
    count=grid.n_bars if expected_bars is None else expected_bars
    if grid.n_bars<count:raise ValueError('Annotation has fewer bars than requested')
    midi=pretty_midi.PrettyMIDI(str(midi_path));raw=mido.MidiFile(str(midi_path))
    first_tick=midi.time_to_tick(grid.beat_times[0]);ppq=raw.ticks_per_beat
    max_error=0.
    for beat,t in enumerate(grid.beat_times[:grid.bar_indices[count]+1]):
        expected_tick=first_tick+beat*ppq
        delta=abs(float(midi.tick_to_time(int(round(expected_tick))))-t)
        max_error=max(max_error,delta)
        if delta>tolerance:raise ValueError(f'Exported MIDI beat clock differs from annotation by {delta:.6f}s')
    signatures=sorted(midi.time_signature_changes,key=lambda x:x.time)
    for bar in range(count):
        active=[s for s in signatures if s.time<=grid.bar_times[bar]+tolerance]
        if not active or active[-1].numerator!=grid.bar_beats[bar] or active[-1].denominator!=4:
            raise ValueError(f'Exported MIDI meter differs from annotation at bar{bar}')
    endpoint=grid.bar_times[count]
    if abs(raw.length-endpoint)>tolerance:
        raise ValueError(f'MIDI endpoint {raw.length:.6f}s differs from {count}bar annotation endpoint {endpoint:.6f}s')
    return dict(n_bars=count,max_beat_time_error_seconds=max_error,midi_duration_seconds=raw.length,
                annotation_duration_seconds=endpoint,quarter_note_semantics='one annotation beat, n/4 meter')


def evaluate_bonus_dir(gen_midi_dir,prompt_dir,output_dir,tag='bonus'):
    gen_midi_dir,prompt_dir,output_dir=Path(gen_midi_dir),Path(prompt_dir),Path(output_dir)
    expected=[f'{i:03d}' for i in range(1,11)]
    found=sorted(p.stem for p in gen_midi_dir.glob('*.mid'))
    if found!=expected:raise ValueError(f'Bonus IDs must be exactly001..010, found {found}')
    output_dir.mkdir(parents=True,exist_ok=True);rows=[];identity={};clocks={}
    args=SimpleNamespace(gen_midi_dir=str(gen_midi_dir),prompt_dir=str(prompt_dir))
    for song in expected:
        midi=pretty_midi.PrettyMIDI(str(gen_midi_dir/f'{song}.mid'))
        names=[i.name.strip().upper() for i in midi.instruments]
        if names.count('MELODY')!=1 or names.count('PIANO')!=1 or len(names)!=2:
            raise ValueError(f'{song}: require exactly named MELODY and PIANO tracks, found {names}')
        ref=pretty_midi.PrettyMIDI(str(prompt_dir/song/'melody.mid'))
        melody=next(i for i in midi.instruments if i.name.strip().upper()=='MELODY')
        identity[song]=check_note_identity([n for i in ref.instruments for n in i.notes],melody.notes)
        clocks[song]=validate_annotation_clock(gen_midi_dir/f'{song}.mid',prompt_dir/song/'beat_midi.txt',16)
        endpoint=clocks[song]['annotation_duration_seconds']
        piano=next(i for i in midi.instruments if i.name.strip().upper()=='PIANO')
        if not piano.notes or any(n.start < -.003 or n.start >= endpoint or n.end > endpoint+.003 for n in piano.notes):
            raise ValueError(f'{song}: empty piano or notes beyond the16bar grid')
        # Apple Accelerate may set spurious IEEE flags in small nonnegative matmul.
        # The unchanged implementation was checked against elementwise sums
        # (500 cases, max abs error6.94e-17); final metrics must still be finite.
        with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
            row=dict(song_name=song,**BONUS.evaluate_song(song,args))
        if row['n_bars']!=16:raise ValueError(f'{song}: {row["n_bars"]} bars, expected16')
        if not all(np.isfinite(row[c]) for c in ['n_bars']+BONUS.METRIC_COLS):raise ValueError(f'{song}: nonfinite metric')
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(output_dir/f'{tag}_per_piece.csv',index=False)
    pooled={c:float(np.average(frame[c],weights=frame.n_bars)) for c in BONUS.METRIC_COLS}
    result=dict(tag=tag,expected_pieces=10,evaluated_pieces=10,skipped_pieces=0,total_bars=160,
                pooled=pooled,summary=metric_summary(rows),melody_identity=identity,annotation_clocks=clocks,
                source_sha256=hashlib.sha256((ROOT/'HW2_scripts/eval_bonus.py').read_bytes()).hexdigest(),
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output_dir/f'{tag}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def compare_to_reference(generated_summary,reference_summary):
    """Distance, not maximization: equal-weight H4/GS absolute standardized gaps."""
    gaps={}
    for metric in ('H4','GS'):
        gen=generated_summary['summary'][metric];ref=reference_summary['summary'][metric]
        gaps[metric]=abs(gen['mean']-ref['mean'])/max(ref['std'],1e-6)
    return dict(standardized_absolute_gaps=gaps,mean_gap=float(np.mean(list(gaps.values()))),
                selection_rule='Minimize mean absolute standardized gap to real H4/GS; inspect degeneration diagnostics separately')


def audit_basic_manifest(manifest_path, output_dir, expected_count=20, expected_bars=32, prompt_dir=None):
    """Independently verify saved files, their clock, and supplied prompt notes."""
    manifest_path,output_dir=Path(manifest_path),Path(output_dir)
    manifest=json.loads(manifest_path.read_text())
    records=manifest['samples']
    if not manifest.get('complete') or len(records)!=expected_count:
        raise ValueError('Manifest is incomplete or has an unexpected sample count')
    names=[r['song_name'] for r in records]
    expected_names=([f'{i:03d}' for i in range(1,4)] if manifest['task']=='continuation'
                    else [f'sample_{i:03d}' for i in range(1,expected_count+1)])
    if sorted(names)!=expected_names:
        raise ValueError(f'Unexpected sample IDs: {names}')
    actual_files=sorted(p.stem for p in manifest_path.parent.glob('*.mid'))
    if actual_files!=sorted(names):
        raise ValueError('Manifest IDs and on-disk MIDI IDs differ')
    audits=[]
    for record in records:
        path=manifest_path.parent/f'{record["song_name"]}.mid'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=record['midi_sha256']:
            raise ValueError(f'{path}: MIDI hash differs from generation manifest')
        bars=parse_bars(record['tokens']);times=np.asarray(record['bar_times'],dtype=float)
        if (len(bars)!=expected_bars or record['bars']!=expected_bars
                or len(times)!=expected_bars+1 or not np.isfinite(times).all()
                or not (np.diff(times)>0).all()):
            raise ValueError(f'{path}: invalid bar count or clock')
        if [b['meter'] for b in bars]!=record['bar_beats']:
            raise ValueError(f'{path}: manifest meter and token meter differ')
        midi=pretty_midi.PrettyMIDI(str(path));raw=mido.MidiFile(str(path))
        notes=[n for inst in midi.instruments for n in inst.notes]
        expected_notes=sum(len(b['notes']) for b in bars)
        if len(notes)!=expected_notes:
            raise ValueError(f'{path}: MIDI/token note counts differ ({len(notes)}/{expected_notes})')
        if not notes or any(n.start<-.003 or n.start>=times[-1]+.003 or n.end>times[-1]+.003 for n in notes):
            raise ValueError(f'{path}: notes absent or beyond the claimed endpoint')
        if abs(raw.length-times[-1])>.004:
            raise ValueError(f'{path}: silent final duration was not retained')
        ppq=raw.ticks_per_beat;first_tick=midi.time_to_tick(float(times[0]));beats=0
        max_error=0.
        for i,t in enumerate(times):
            delta=abs(float(midi.tick_to_time(int(round(first_tick+beats*ppq))))-t)
            max_error=max(max_error,delta)
            if delta>.004:raise ValueError(f'{path}: MIDI bar boundary differs by {delta:.6f}s')
            if i<len(bars):
                active=[s for s in midi.time_signature_changes if s.time<=t+.004]
                if not active or active[-1].numerator!=bars[i]['meter'] or active[-1].denominator!=4:
                    raise ValueError(f'{path}: MIDI time signature differs from annotation-derived token meter')
                beats+=bars[i]['meter']
        row=dict(song_name=record['song_name'],midi=str(path),midi_sha256=digest,n_bars=len(bars),
                 token_notes=expected_notes,midi_notes=len(notes),duration_seconds=raw.length,
                 max_bar_boundary_error_seconds=max_error)
        if manifest['task']=='continuation':
            if prompt_dir is None:raise ValueError('Continuation audit requires the original prompt directory')
            folder=Path(prompt_dir)/record['song_name']
            prompt_files=list(folder.glob('*.mid'))
            if len(prompt_files)!=1:raise ValueError('Expected exactly one original prompt MIDI')
            row['prefix_identity']=validate_continuation(prompt_files[0],path,folder/'beat_midi.txt')
        audits.append(row)
    result=dict(manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                expected_pieces=expected_count,evaluated_pieces=len(audits),skipped_pieces=0,
                expected_bars=expected_bars,files=audits,
                human_listening_performed=False,
                note='Bar boundaries and n/4 signatures are defined by the annotation-derived clock; original source MIDI meter is not authoritative.')
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'manifest_integrity_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('basic');inputs=a.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--sequences',type=Path);inputs.add_argument('--manifest',type=Path)
    a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--prompt-dir',type=Path)
    a.add_argument('--expected-count',type=int,default=20);a.add_argument('--expected-bars',type=int,default=32);a.add_argument('--tag',default='basic')
    b=sub.add_parser('bonus');b.add_argument('--gen-midi-dir',type=Path,required=True);b.add_argument('--prompt-dir',type=Path,required=True)
    b.add_argument('--output-dir',type=Path,required=True);b.add_argument('--tag',default='bonus')
    args=p.parse_args()
    if args.task=='basic':
        if args.manifest:
            audit_basic_manifest(args.manifest,args.output_dir,args.expected_count,args.expected_bars,args.prompt_dir)
            manifest=json.loads(args.manifest.read_text());records=manifest['samples']
        else:records=json.loads(args.sequences.read_text())
        result=evaluate_basic_records(records,args.output_dir,args.expected_count,args.expected_bars,args.tag)
        if args.manifest:
            result['generation_provenance']={k:manifest[k] for k in ('task','checkpoint','checkpoint_sha256','configuration')}
            result['generation_provenance']['manifest_sha256']=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
            (args.output_dir/f'{args.tag}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    else:result=evaluate_bonus_dir(args.gen_midi_dir,args.prompt_dir,args.output_dir,args.tag)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
