"""Additional objective diagnostics; these do not claim a human listening test.

Build a fixed real-data reference from pre-refit validation songs; measure exact local training
matches, density, rest/repetition and continuation boundary behavior. A lack of
exact training matches is not proof of non-memorization or good musical quality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate import parse_bars, _pattern, evaluate_basic_records


def load_token_archive(path,metadata_path,vocabulary_path):
    archive=np.load(path,allow_pickle=False)
    meta=json.loads(Path(metadata_path).read_text())
    vocab=json.loads(Path(vocabulary_path).read_text())['tokens']
    def song(i):
        lo,hi=archive['offsets'][i:i+2]
        return [vocab[int(t)] for t in archive['tokens'][lo:hi]]
    return archive,meta,vocab,song


def build_real_reference(archive_path,metadata_path,vocab_path,output_dir,split='val',bars=32):
    """One deterministic center-nearest eligible crop/song; no metric-based selection."""
    data,meta,vocab,song=load_token_archive(archive_path,metadata_path,vocab_path)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    records=[];selection=[];excluded=[]
    for i,item in enumerate(meta):
        if split!='all' and str(data['splits'][i])!=split:continue
        n=item['n_bars']
        if n<bars:
            excluded.append(dict(song_name=item['song_name'],reason='shorter than32bars',n_bars=n));continue
        valid=[st for st in range(n-bars+1) if max(item['bar_beats'][st:st+bars])<=8]
        if not valid:
            excluded.append(dict(song_name=item['song_name'],reason='no32bar crop representable by official128position grid',n_bars=n));continue
        center=(n-bars)/2
        start=min(valid,key=lambda st:(abs(st-center),st))
        b0,b1=data['bar_song_offsets'][i:i+2]
        offsets=data['bar_offsets'][b0:b1]
        seq=song(i)
        cropped=[t for t in seq[int(offsets[start]):int(offsets[start+bars])] if t not in ('<BOS>','<EOS>')]
        record=dict(song_name=item['song_name'],tokens=['<BOS>']+cropped+['<EOS>'])
        if len(parse_bars(record['tokens']))!=bars:raise ValueError('Archive crop boundary mismatch')
        records.append(record);selection.append(dict(song_name=item['song_name'],start_bar=start,n_bars=bars,
                                                    source_bars=n,split=str(data['splits'][i])))
    if not records:raise ValueError('No eligible real reference pieces')
    (output_dir/'real_reference_sequences.json').write_text(json.dumps(records)+'\n')
    scope=('Pre-refit validation split, held out from the diagnostic model only. The final model is refit on all896 official training songs, including these songs, so this is a descriptive real-data reference and is not independent of that final model.'
           if split=='val' else f'Eligible official training songs from corpus split={split}. This descriptive real-data reference is not independent of the final model, which is refit on all896 official training songs.')
    selection_record=dict(rule=f'One center-nearest32bar crop per eligible song in corpus split={split}, ties choose earlier; no metric-based selection',
                          split=split,included=len(records),excluded=excluded,selection=selection,
                          reference_scope=scope,
                          rare_meter_policy='12beat bars cannot fit official128position16subdivision grid; choose nearest eligible crop')
    (output_dir/'real_reference_selection.json').write_text(json.dumps(selection_record,indent=2)+'\n')
    result=evaluate_basic_records(records,output_dir,len(records),bars,'real_reference')
    result['reference_scope']=selection_record['reference_scope']
    result['reference_split']=split
    result['reference_selection_counts']=dict(included=len(records),excluded=len(excluded),requested_split_songs=len(records)+len(excluded))
    result['selection_file']='real_reference_selection.json'
    result['reference_source_sha256']={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                       for p in (archive_path,metadata_path,vocab_path)}
    result['reference_builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output_dir/'real_reference_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def bar_hash(bar):
    return hashlib.sha256(json.dumps(_pattern(bar),separators=(',',':')).encode()).hexdigest()[:24]


def build_copy_index(archive_path,metadata_path,vocab_path,widths=(1,2,4,8),split='train'):
    data,meta,vocab,song=load_token_archive(archive_path,metadata_path,vocab_path)
    index={width:{} for width in widths}
    for i,item in enumerate(meta):
        # Final all-song refits include the former validation songs as well.
        if split!='all' and str(data['splits'][i])!=split:continue
        hashes=[bar_hash(b) for b in parse_bars(song(i))]
        for width in widths:
            for j in range(len(hashes)-width+1):
                key=tuple(hashes[j:j+width])
                if key not in index[width]:index[width][key]=(item['song_name'],j)
    return index


def training_copy_analysis(records,index,output_dir,tag='copy',reference='Pre-refit optimization training split only, all bars'):
    rows=[];full_hashes=[]
    for record in records:
        bars=parse_bars(record['tokens']);hashes=[bar_hash(b) for b in bars]
        full_hashes.append(tuple(hashes))
        row={'song_name':record['song_name']}
        for width,lookup in index.items():
            matches=[(j,lookup[tuple(hashes[j:j+width])]) for j in range(len(hashes)-width+1)
                     if tuple(hashes[j:j+width]) in lookup]
            row[f'exact_{width}bar_match_fraction']=len(matches)/max(1,len(hashes)-width+1)
            row[f'exact_{width}bar_match_count']=len(matches)
            nonempty={j for j in range(len(hashes)-width+1) if any(b['notes'] for b in bars[j:j+width])}
            nonempty_matches=[m for m in matches if m[0] in nonempty]
            row[f'exact_{width}bar_nonempty_match_fraction']=len(nonempty_matches)/max(1,len(nonempty))
            row[f'exact_{width}bar_nonempty_match_count']=len(nonempty_matches)
            row[f'{width}bar_empty_window_count']=max(0,len(hashes)-width+1)-len(nonempty)
            if matches:row[f'first_{width}bar_match']=str(matches[0])
        rows.append(row)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows);frame.to_csv(output_dir/f'{tag}_per_piece.csv',index=False)
    result=dict(n_pieces=len(rows),duplicate_full_piece_count=len(full_hashes)-len(set(full_hashes)),
                mean_match_fractions={c:float(frame[c].mean()) for c in frame if c.endswith('_fraction')},
                definition='Exact contiguous bars with same position8/pitch/duration8 multisets; tempo, velocity and track names ignored. Raw match fractions include silent windows; nonempty fractions exclude entirely silent windows from numerator and denominator.',
                reference=reference,
                limitation='This conservative exact-match test does not detect approximate or transposed copying; zero matches does not establish originality or quality')
    (output_dir/f'{tag}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def boundary_analysis(tokens,boundary_bar=8,window=2):
    bars=parse_bars(tokens)
    if len(bars)<boundary_bar+window:raise ValueError('Insufficient continuation bars')
    def describe(sub):
        notes=[n for b in sub for n in b['notes']]
        return dict(notes_per_bar=len(notes)/len(sub),pitch_mean=float(np.mean([n['pitch'] for n in notes])) if notes else None,
                    velocity_mean=float(np.mean([n['velocity'] for n in notes])) if notes else None,
                    duration_beats_mean=float(np.mean([n['duration']/8 for n in notes])) if notes else None,
                    pitch_classes=sorted(set(n['pitch']%12 for n in notes)))
    before=describe(bars[boundary_bar-window:boundary_bar]);after=describe(bars[boundary_bar:boundary_bar+window])
    a,b=set(before['pitch_classes']),set(after['pitch_classes'])
    return dict(before=before,after=after,pitch_class_jaccard=len(a&b)/len(a|b) if a|b else 1.,
                pitch_mean_jump=None if before['pitch_mean'] is None or after['pitch_mean'] is None else after['pitch_mean']-before['pitch_mean'],
                note_density_ratio=after['notes_per_bar']/max(before['notes_per_bar'],1e-9),
                interpretation='Descriptive boundary continuity only; abrupt musical changes can be valid')


def paired_metric_comparison(first_rows,second_rows,metrics=('H4','GS','GS_active'),seed=6221):
    """Paired piece/seed bootstrap; caller must provide identical pairing IDs."""
    first={str(r['song_name']):r for r in first_rows};second={str(r['song_name']):r for r in second_rows}
    if len(first)!=len(first_rows) or len(second)!=len(second_rows) or set(first)!=set(second):
        raise ValueError('Paired comparison requires unique and identical piece IDs (same generation seed schedule)')
    ids=sorted(first);rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(ids),(5000,len(ids)));result={}
    for metric in metrics:
        diff=np.array([second[i][metric]-first[i][metric] for i in ids],dtype=float)
        if not np.isfinite(diff).all():raise ValueError('Nonfinite paired metric')
        means=diff[draws].mean(axis=1)
        result[metric]=dict(mean_second_minus_first=float(diff.mean()),
                            paired_mean_ci95=np.quantile(means,[.025,.975]).tolist(),n_pairs=len(ids))
    return dict(metrics=result,method='5000 whole-piece paired bootstrap resamples with fixed seed6221',
                caution='Same seeds reduce sampling variability but do not imply note-level correspondence across configurations; this is not an official grading score')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('reference');a.add_argument('--archive',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True)
    a.add_argument('--vocabulary',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--split',choices=['val','train','all'],default='val')
    b=sub.add_parser('copy');b.add_argument('--archive',type=Path,required=True);b.add_argument('--metadata',type=Path,required=True)
    b.add_argument('--vocabulary',type=Path,required=True);b.add_argument('--sequences',type=Path,required=True)
    b.add_argument('--output-dir',type=Path,required=True);b.add_argument('--tag',default='copy')
    b.add_argument('--split',choices=['val','train','all'],default='train')
    args=p.parse_args()
    if args.task=='reference':result=build_real_reference(args.archive,args.metadata,args.vocabulary,args.output_dir,args.split)
    else:
        records=json.loads(args.sequences.read_text())
        if isinstance(records,dict):records=records['samples']
        # Fail on an incompatible (e.g. already canonicalized) representation
        # before spending time indexing the complete training corpus.
        for record in records:parse_bars(record['tokens'])
        result=training_copy_analysis(records,build_copy_index(args.archive,args.metadata,args.vocabulary,split=args.split),
                                      args.output_dir,args.tag,f'All bars from corpus split={args.split}; all includes all896 final-refit training songs')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
