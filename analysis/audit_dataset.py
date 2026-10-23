#!/usr/bin/env python3
"""Read-only POP909 audit; run from project root:

    hw1/.venv/bin/python hw2/analysis/audit_dataset.py

Only Python standard library is required. Writes data_audit.json and
summary-only data_summary.json beside this script; original inputs are unchanged.
"""
from pathlib import Path
from collections import Counter,defaultdict,deque
import struct,hashlib,json,bisect,statistics,time
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'POP909'; OUT=ROOT/'analysis/data_audit.json'

def varlen(d,i):
    n=0
    for _ in range(5):
        v=d[i];i+=1;n=(n<<7)|(v&127)
        if not(v&128):return n,i
    raise ValueError('bad VLQ')

def parse_midi(path):
    d=path.read_bytes(); assert d[:4]==b'MThd'
    h=struct.unpack('>I',d[4:8])[0];fmt,ntr,tpq=struct.unpack('>HHH',d[8:14]); assert tpq<32768
    i=8+h; tracks=[];tempos=[];meters=[];allnotes=[]
    for tr in range(ntr):
        assert d[i:i+4]==b'MTrk'; ln=struct.unpack('>I',d[i+4:i+8])[0];t=d[i+8:i+8+ln];i+=8+ln
        j=0;tick=0; running=None;name='';pending=defaultdict(deque);notes=[];programs=[];events=Counter();max_tick=0;unmatched_end=0
        while j<len(t):
            delta,j=varlen(t,j);tick+=delta;status=t[j]
            if status<128:
                assert running is not None;status=running
            else:
                j+=1
            if status==255:
                ty=t[j];j+=1;sz,j=varlen(t,j);blob=t[j:j+sz];j+=sz;events[f'meta_{ty}']+=1
                if ty==3:name=blob.decode('utf8',errors='replace')
                if ty==81:tempos.append((tick,int.from_bytes(blob,'big')))
                if ty==88:meters.append((tick,blob[0],2**blob[1],blob[2],blob[3]))
                if ty==47:max_tick=max(max_tick,tick)
                continue
            if status in (240,247):
                sz,j=varlen(t,j);j+=sz;continue
            assert status<240
            running=status;ty=status>>4;ch=status&15
            size=1 if ty in(12,13) else 2;vs=t[j:j+size];j+=size;events[f'ch_{ty}']+=1
            if ty==12:programs.append((tick,ch,vs[0]))
            if ty==9 and vs[1]>0:pending[(ch,vs[0])].append((tick,vs[1]))
            elif ty==8 or (ty==9 and vs[1]==0):
                if pending[(ch,vs[0])]:
                    start,vel=pending[(ch,vs[0])].popleft();notes.append((start,tick,vs[0],vel,ch,tr))
                else:unmatched_end+=1
            max_tick=max(max_tick,tick)
        allnotes.extend(notes)
        tracks.append({'name':name,'notes':len(notes),'pitch_range':[min(n[2] for n in notes),max(n[2] for n in notes)] if notes else None,'programs':programs,'channels':sorted({n[4] for n in notes}),'end_tick':max_tick,'unclosed_notes':sum(map(len,pending.values())),'unmatched_note_offs':unmatched_end})
    tempo_map=dict([(0,500000)]+sorted(tempos));tempo_ticks=sorted(tempo_map);seconds=[0.]
    for a,b in zip(tempo_ticks,tempo_ticks[1:]):seconds.append(seconds[-1]+(b-a)*tempo_map[a]/1e6/tpq)
    def secs(t):
        k=bisect.bisect_right(tempo_ticks,t)-1
        return seconds[k]+(t-tempo_ticks[k])*tempo_map[tempo_ticks[k]]/1e6/tpq
    bytr={}
    for k,tr in enumerate(tracks):
        nn=[n for n in allnotes if n[5]==k]
        if nn:bytr[tr['name'] or f'track_{k}']={'notes':len(nn),'first_note_sec':secs(min(n[0] for n in nn)),'last_note_end_sec':secs(max(n[1] for n in nn)),'overlapping_same_pitch':sum(1 for a,b in zip(sorted(nn,key=lambda n:(n[4],n[2],n[0])), sorted(nn,key=lambda n:(n[4],n[2],n[0]))[1:]) if a[4]==b[4] and a[2]==b[2] and a[1]>b[0])}
    # Exact music fingerprint independent of metadata and PPQ, with source track names and velocities.
    fp=hashlib.sha256(json.dumps(sorted((tracks[n[5]]['name'],round(n[0]/tpq,8),round(n[1]/tpq,8),n[2],n[3]) for n in allnotes),separators=(',',':')).encode()).hexdigest()
    return {'format':fmt,'ticks_per_quarter':tpq,'track_count':ntr,'tracks':tracks,'note_count':len(allnotes),'duration_sec':secs(max(t['end_tick'] for t in tracks)),'note_end_sec':secs(max(n[1] for n in allnotes)) if allnotes else 0,'tempo_events':len(tempos),'tempo_qpm_range':[min(60e6/x[1] for x in tempos),max(60e6/x[1] for x in tempos)] if tempos else [120,120],'time_signatures':sorted(set(f'{n}/{den}' for _,n,den,_,_ in meters)), 'meter_events':meters, 'track_note_timing':bytr, 'note_fingerprint':fp},allnotes,secs

def read_txt(path):
    return [l.split() for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]

rows=[];fail=[];sha_groups=defaultdict(list);full_fp=defaultdict(list);training_song_ids=[];prompt_details=[]
start=time.time()
for sub in ['training','prompts/basic-b','prompts/bonus']:
  for folder in sorted((DATA/sub).iterdir()):
    if not folder.is_dir():continue
    name=f'{sub}/{folder.name}';files=sorted(folder.glob('*.mid'));beats=read_txt(folder/'beat_midi.txt');chords=read_txt(folder/'chord_midi.txt')
    times=[float(r[0]) for r in beats];db=[i for i,r in enumerate(beats) if len(r)>2 and float(r[2])==1];intervals=[b-a for a,b in zip(db,db[1:])];grid_sec=statistics.median(b-a for a,b in zip(times,times[1:]))
    beat={'rows':len(beats),'columns':sorted(set(map(len,beats))),'first_sec':times[0],'last_sec':times[-1],'strictly_increasing':all(b>a for a,b in zip(times,times[1:])),'downbeats':len(db),'opening_beats_before_first_downbeat':db[0] if db else None,'downbeat_index_intervals':dict(Counter(intervals)),'second_col_values':sorted({r[1] for r in beats}),'third_col_values':sorted({r[2] for r in beats}),'beat_duration_median_sec':grid_sec,'beat_duration_range_sec':[min(b-a for a,b in zip(times,times[1:])),max(b-a for a,b in zip(times,times[1:]))],'extrapolated_end_sec':times[-1]+(times[-1]-times[-2])}
    chord={'segments':len(chords),'labels':sorted({r[2] for r in chords}),'first_sec':float(chords[0][0]),'last_end_sec':float(chords[-1][1]),'gaps_or_overlaps':sum(abs(float(a[1])-float(b[0]))>1e-5 for a,b in zip(chords,chords[1:])),'nonpositive_segments':sum(float(r[1])<=float(r[0]) for r in chords)}
    rec={'item':name,'beat':beat,'chord':chord,'midis':{}}
    for path in files:
      try:
        midi,notes,secs=parse_midi(path);rec['midis'][path.name]=midi;full_fp[midi['note_fingerprint']].append(f'{name}/{path.name}')
        downbeat_times=[times[k] for k in db]
        bar_note_counts=Counter(max(0,bisect.bisect_right(downbeat_times,secs(n[0]))-1)+(1 if db and db[0]>0 else 0) for n in notes)
        n_bars=max(bar_note_counts)+1 if bar_note_counts else 0
        bar_counts=[bar_note_counts[k] for k in range(n_bars)]
        midi['bar_note_counts']=bar_counts
        midi['estimated_note_tokens']=4*len(notes)+n_bars+2+2*midi['tempo_events']
        midi['window_token_estimates']={str(width):[4*sum(bar_counts[k:k+width])+width+4 for k in range(0,max(0,n_bars-width+1),max(1,width//2))] for width in [8,16,32]}

        for note in notes:
            if note[1]<=note[0]:fail.append({'item':name,'problem':'nonpositive_note','note':note})
        for tr,timing in midi['track_note_timing'].items():
            timing['note_onsets_before_first_beat']=sum(secs(n[0])<times[0]-1e-5 for n in notes if midi['tracks'][n[5]]['name']==tr)
            timing['note_onsets_at_or_after_extrapolated_end']=sum(secs(n[0])>=beat['extrapolated_end_sec']-1e-5 for n in notes if midi['tracks'][n[5]]['name']==tr)
      except Exception as e:fail.append({'item':str(path),'error':repr(e)})
    for f in folder.iterdir():
      if f.is_file():sha_groups[(f.suffix,hashlib.sha256(f.read_bytes()).hexdigest())].append(str(f.relative_to(DATA)))
    rows.append(rec)
    if sub=='training':training_song_ids.append(folder.name)
    else:prompt_details.append(rec)

training=[r for r in rows if r['item'].startswith('training/')]
def rng(xs):return {'min':min(xs),'median':statistics.median(xs),'max':max(xs)}
training_midis=[v for r in training for v in r['midis'].values()]
summary={'counts':dict(Counter(r['item'].rsplit('/',1)[0] for r in rows)),'training_ids_missing_from_001_909':[f'{i:03}' for i in range(1,910) if f'{i:03}' not in training_song_ids],'training_midi_count':len(training_midis),'training_track_name_combinations':dict(Counter('|'.join(t['name'] for t in m['tracks']) for m in training_midis)),'training_time_signature_sets':dict(Counter('|'.join(m['time_signatures']) for m in training_midis)),'training_duration_sec':rng([m['duration_sec'] for m in training_midis]),'training_note_count':rng([m['note_count'] for m in training_midis]),'training_note_total':sum(m['note_count'] for m in training_midis),'training_tempo_qpm_min':min(m['tempo_qpm_range'][0] for m in training_midis),'training_tempo_qpm_max':max(m['tempo_qpm_range'][1] for m in training_midis),'training_downbeat_interval_counts':dict(sum((Counter({int(k):v for k,v in r['beat']['downbeat_index_intervals'].items()}) for r in training),Counter())),'midi_parse_or_note_failures':fail,'nonmonotonic_beat_files':[r['item'] for r in rows if not r['beat']['strictly_increasing']],'unclosed_or_unmatched_notes':[{'item':r['item'],'midi':p,'track':t['name'],'unclosed':t['unclosed_notes'],'unmatched':t['unmatched_note_offs']} for r in rows for p,m in r['midis'].items() for t in m['tracks'] if t['unclosed_notes'] or t['unmatched_note_offs']],'duplicate_fullfile_hash_groups':[{'suffix':k[0],'sha256':k[1],'files':v} for k,v in sha_groups.items() if len(v)>1],'duplicate_complete_note_fingerprint_groups':[{'sha256':k,'files':v} for k,v in full_fp.items() if len(v)>1],'elapsed_sec':time.time()-start}
def pct(xs):
    xs=sorted(xs)
    return {str(p):xs[min(len(xs)-1,int((len(xs)-1)*p/100))] for p in [0,10,25,50,75,90,95,99,100]}
summary['training_bars_per_song']=pct([len(m['bar_note_counts']) for m in training_midis])
summary['training_note_tokens_per_song']=pct([m['estimated_note_tokens'] for m in training_midis])
summary['training_notes_per_bar']=pct([n for m in training_midis for n in m['bar_note_counts']])
summary['training_window_token_estimates']={str(w):pct([n for m in training_midis for n in m['window_token_estimates'][str(w)]]) for w in [8,16,32]}
summary['training_window_token_exceedance']={str(w):{str(limit):sum(v>limit for v in values)/len(values) for limit in [1024,2048,4096,8192]} for w in [8,16,32] for values in [[n for m in training_midis for n in m['window_token_estimates'][str(w)]]]}
summary['training_pitch_range']=[min(t['pitch_range'][0] for m in training_midis for t in m['tracks'] if t['notes']),max(t['pitch_range'][1] for m in training_midis for t in m['tracks'] if t['notes'])]
summary['training_chord_vocabulary']=sorted({c for r in training for c in r['chord']['labels']})
summary['training_chord_vocabulary_size']=len(summary['training_chord_vocabulary'])
summary['training_with_pickup_before_first_downbeat']=sum(r['beat']['opening_beats_before_first_downbeat']>0 for r in training)
summary['training_with_multiple_annotated_bar_lengths']=sum(len(r['beat']['downbeat_index_intervals'])>1 for r in training)
out={'token_estimate_method':'Each note uses four tokens (Pos, Pitch, Velocity, Duration); every annotation-derived bar adds one Bar token; BOS/EOS add two tokens separately. Full-song estimate adds two tokens per MIDI tempo event. Window estimate assumes one tempo event (two tokens). Track/chord/conditioning tokens are not included. Bar indices follow downbeat markers with a separate pickup bar; end-of-song silent bars may be excluded, matching the reference tokenizer note-driven final bar. No special token is counted as a bar.', 'method':'Actual byte-level Standard MIDI File parse of all MIDI files and all supplied beat/chord text files, Python standard library; independent of pretty_midi. No model trained. Note fingerprints include metadata-independent track names, beat-relative note timing, pitch and velocity; fullfile hashes do not rule out overlapping excerpts.','summary':summary,'prompt_details':prompt_details,'all_items':rows}
OUT.write_text(json.dumps(out,indent=2)+'\n')
summary_out={'method':out['method'],'token_estimate_method':out['token_estimate_method'],'summary':summary,'prompts':[]}
for r in prompt_details:
    summary_out['prompts'].append({'item':r['item'],'bars':r['beat']['downbeats']-1,'beats':r['beat']['rows']-1,'duration_sec':r['beat']['last_sec'],'bar_beat_counts':r['beat']['downbeat_index_intervals'],'has_final_boundary_beat':r['beat']['third_col_values'] and abs(r['beat']['last_sec']-r['chord']['last_end_sec'])<1e-5,'midi_metadata':{fn:{'time_signatures':m['time_signatures'],'tempo_qpm_range':m['tempo_qpm_range'],'tracks':[{key:t[key] for key in ['name','notes','pitch_range']} for t in m['tracks']],'note_end_sec':m['note_end_sec'],'note_token_estimate':m['estimated_note_tokens']} for fn,m in r['midis'].items()}})
(OUT.parent/'data_summary.json').write_text(json.dumps(summary_out,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='training_chord_vocabulary'},indent=2))
print('Saved',OUT)
