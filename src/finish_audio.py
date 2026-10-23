"""Render the selected20 basic-a,9 basic-b and10 formal bonus WAVs.

Wait for completed generation manifests and the saved basic-a selection. This
tool only adds audio metadata to those manifests; MIDI bytes remain unchanged.
Native macOS uses the installed piano bank; FluidSynth requires --soundfont.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import mido
from render_audio import inspect_wav,render_midi

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.audio-tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def musical_manifest(manifest):
    """Ignore only the two record fields that this renderer is allowed to edit."""
    copy=json.loads(json.dumps(manifest))
    for record in copy['samples']:
        record.pop('wav_path',None);record.pop('audio_audit',None)
    return copy


def ready_jobs(root):
    out=root/'results/evaluation';selection_path=out/'selection_progress.json'
    if not selection_path.exists():return None,'Awaiting basic-a configuration selection'
    selection=read(selection_path);selected=selection['selected_configuration']
    if selected not in ('t08','t10','t12','k16','k32'):raise ValueError('Unknown selected configuration')
    specs=[('basic_a',root/f'results/basic_a/{selected}/manifest.json',20)]
    specs += [('basic_b',root/f'results/basic_b/{c}/manifest.json',3) for c in ('t08','t10','t12')]
    specs += [('bonus',root/'results/bonus/manifest.json',10)]
    jobs=[]
    for task,path,count in specs:
        if not path.exists():return None,f'Awaiting {path.relative_to(root)}'
        manifest=read(path)
        if not manifest.get('complete') or len(manifest.get('samples',[]))!=count:
            return None,f'Awaiting all{count} samples in {path.relative_to(root)}'
        expected=([f'sample_{i:03d}' for i in range(1,21)] if task=='basic_a'
                  else [f'{i:03d}' for i in range(1,count+1)])
        if sorted(r['song_name'] for r in manifest['samples'])!=expected:
            raise ValueError(f'{path}: unexpected sample IDs')
        if task=='bonus' and (manifest.get('conditioning',manifest['configuration'].get('conditioning','normal'))!='normal'
                              or manifest.get('submission_eligible') is not True):
            raise ValueError('Formal audio requires normal, submission-eligible bonus generation')
        if task=='basic_a':
            recorded=selection.get('configuration_manifest_sha256',{}).get(selected)
            if recorded and recorded!=digest(path):
                return None,'Awaiting evaluation of the current selected basic-a manifest'
        tag=(selected if task=='basic_a' else path.parent.name if task=='basic_b' else 'bonus')
        evaluation=(out/task/tag/f'{tag}_summary.json' if task!='bonus' else out/'bonus/bonus_summary.json')
        if not evaluation.exists():return None,f'Awaiting strict evaluation of {path.relative_to(root)}'
        summary=read(evaluation)
        if (summary.get('evaluated_pieces')!=count or summary.get('skipped_pieces')!=0
                or summary.get('generation_provenance',{}).get('manifest_sha256')!=digest(path)):
            return None,f'Awaiting current evaluation provenance for {path.relative_to(root)}'
        for record in manifest['samples']:
            midi=path.parent/f'{record["song_name"]}.mid'
            if not midi.exists() or digest(midi)!=record['midi_sha256']:
                raise ValueError(f'{midi}: absent or mismatched MIDI bytes')
            jobs.append((task,path,record['song_name'],record['midi_sha256']))
    if len(jobs)!=39:raise ValueError('Expected exactly39 formal audio files')
    return (selected,jobs),'Ready'


def audit_existing(record,midi,wav,backend,rate,tail):
    audit=record.get('audio_audit',{})
    if (not wav.exists() or audit.get('midi_sha256')!=digest(midi)
            or audit.get('wav_sha256')!=digest(wav) or audit.get('sample_rate')!=rate
            or backend!='auto' and audit.get('backend')!=backend):return None
    observed=inspect_wav(wav)
    if observed['channels']!=2 or observed['clipped_fraction']>1e-4:return None
    if audit.get('backend')=='native' and abs(observed['duration_seconds']-(mido.MidiFile(str(midi)).length+tail))>2/rate:
        return None
    return {**audit,**observed}


def finish(root,selected,jobs,args,status_path):
    files=[];rendered=0;reused=0;baselines={}
    for _,path,_,_ in jobs:
        if path not in baselines:baselines[path]=musical_manifest(read(path))
    for task,path,song,midi_sha in jobs:
        manifest=read(path)
        if musical_manifest(manifest)!=baselines[path]:raise ValueError('Generation manifest changed during audio finishing')
        record=next(r for r in manifest['samples'] if r['song_name']==song)
        midi=path.parent/f'{song}.mid';wav=midi.with_suffix('.wav')
        if digest(midi)!=midi_sha:raise ValueError('MIDI changed before audio render')
        audit=audit_existing(record,midi,wav,args.backend,args.sample_rate,args.tail_seconds)
        if audit is None:
            audit=render_midi(midi,wav,args.sample_rate,args.tail_seconds,args.backend,args.soundfont);rendered+=1
        else:reused+=1
        if digest(midi)!=midi_sha or audit['midi_sha256']!=midi_sha:
            raise ValueError('Audio rendering changed MIDI bytes')
        if audit['sample_rate']!=args.sample_rate or audit['channels']!=2:
            raise ValueError('Expected stereo audio at the requested sample rate')
        audit['midi']=str(midi.relative_to(root));audit['wav']=str(wav.relative_to(root))
        audit['midi_duration_seconds']=mido.MidiFile(str(midi)).length
        audit['release_tail_seconds']=audit['duration_seconds']-audit['midi_duration_seconds']
        # Read again before the atomic update so unrelated concurrent writes are
        # caught rather than replaced. Evaluation only reads these manifests.
        manifest=read(path)
        if musical_manifest(manifest)!=baselines[path]:raise ValueError('Generation provenance changed during audio render')
        record=next(r for r in manifest['samples'] if r['song_name']==song)
        record['wav_path']=str(wav.relative_to(root));record['audio_audit']=audit
        if musical_manifest(manifest)!=baselines[path]:raise ValueError('Unexpected non-audio manifest edit')
        write(path,manifest)
        files.append({'task':task,'song_name':song,'manifest':str(path.relative_to(root)),**audit})
        partial={'complete':False,'state':'rendering','selected_configuration':selected,'expected_files':39,
                 'completed_files':len(files),'rendered_this_run':rendered,'reused_this_run':reused,
                 'midi_bytes_unchanged':True,'files':files,'failures':[]}
        write(status_path,partial)
        print(json.dumps({'audio_completed':len(files),'expected':39,'wav':audit['wav'],'rms':audit['rms'],'clipped_fraction':audit['clipped_fraction']}),flush=True)
    for path,baseline in baselines.items():
        manifest=read(path)
        if musical_manifest(manifest)!=baseline:raise ValueError('Final manifest musical provenance differs')
        audits=[r['audio_audit'] for r in manifest['samples']]
        write(path.parent/'render_manifest.json',{'count':len(audits),'expected_count':len(manifest['samples']),
              'complete':True,'failures':0,'files':audits,'midi_bytes_unchanged':True})
    counts={task:sum(r['task']==task for r in files) for task in ('basic_a','basic_b','bonus')}
    if counts!={'basic_a':20,'basic_b':9,'bonus':10}:raise ValueError('Final audio task counts differ')
    result={'complete':True,'state':'complete','selected_configuration':selected,'expected_files':39,
            'completed_files':39,'counts':counts,'rendered_this_run':rendered,'reused_this_run':reused,
            'midi_bytes_unchanged':True,'manifest_non_audio_fields_unchanged':True,
            'maximum_clipped_fraction':max(r['clipped_fraction'] for r in files),'files':files,'failures':[],
            'scope':'Formal selected basic-a20, all basic-b9, normal bonus10; the empty-melody ablation is excluded.'}
    write(status_path,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--backend',choices=['auto','native','fluidsynth'],default='auto')
    parser.add_argument('--soundfont',type=Path)
    parser.add_argument('--sample-rate',type=int,default=22050)
    parser.add_argument('--tail-seconds',type=float,default=2.)
    parser.add_argument('--poll-seconds',type=float,default=20.)
    parser.add_argument('--deadline-hours',type=float,default=24.)
    parser.add_argument('--once',action='store_true',help='Exit if required generation files are not ready')
    args=parser.parse_args();root=args.root.resolve();out=root/'results/evaluation';out.mkdir(parents=True,exist_ok=True)
    # All supported native/FluidSynth deployment environments are Unix here.
    import fcntl
    lock=(out/'finish_audio.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another audio finisher is already active')
    status_path=out/'finish_audio_status.json';deadline=time.monotonic()+args.deadline_hours*3600;last_message=None
    while time.monotonic()<deadline:
        try:
            ready,message=ready_jobs(root)
            if ready:
                selected,jobs=ready;finish(root,selected,jobs,args,status_path)
                print('ALL39_FORMAL_WAVS_COMPLETE',flush=True);return
            write(status_path,{'complete':False,'state':'waiting','reason':message,'expected_files':39,'failures':[]})
            if message!=last_message:print(message,flush=True);last_message=message
            if args.once:raise SystemExit(2)
        except Exception:
            failure=traceback.format_exc()
            write(status_path,{'complete':False,'state':'failed','expected_files':39,'failures':[failure]})
            print(failure,flush=True);raise
        time.sleep(args.poll_seconds)
    raise SystemExit('Audio finishing deadline reached before all files were ready')


if __name__=='__main__':main()
