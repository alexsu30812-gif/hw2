"""Evaluate only complete genuine experiment manifests; never generate music."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time
import traceback
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from evaluate import (audit_basic_manifest,evaluate_basic_records,evaluate_bonus_dir,
                      compare_to_reference,piece_diagnostics,parse_bars,BONUS)
from listening_analysis import training_copy_analysis,boundary_analysis,paired_metric_comparison

BASE=ROOT/'results';OUT=BASE/'evaluation'
A=('t08','t10','t12','k16','k32');B=('t08','t10','t12')
REF=json.loads((OUT/'real_reference/real_reference_summary.json').read_text())
with (OUT/'cache/copy_indices.pkl').open('rb') as f:INDICES=pickle.load(f)
done=(json.loads((OUT/'watch_status.json').read_text()).get('complete_manifests_evaluated',{})
      if (OUT/'watch_status.json').exists() else {})
errors={}


def read(path):return json.loads(path.read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def evaluate_manifest(path,kind,config,count):
    manifest=read(path)
    if not manifest.get('complete') or len(manifest.get('samples',[]))!=count:return
    digest=hashlib.sha256(path.read_bytes()).hexdigest();key=str(path)
    if done.get(key)==digest:return
    directory=OUT/kind/config if kind in ('basic_a','basic_b') else OUT/config
    prompt_dir=ROOT/'POP909/prompts/basic-b' if kind=='basic_b' else None
    audit_basic_manifest(path,directory,count,32,prompt_dir)
    summary=evaluate_basic_records(manifest['samples'],directory,count,32,config)
    summary['generation_provenance']={k:manifest[k] for k in ('task','checkpoint','checkpoint_sha256','configuration')}
    summary['generation_provenance']['manifest_sha256']=digest
    write(directory/f'{config}_summary.json',summary)
    if kind!='basic_b':
        split='all' if kind=='basic_a' else 'train'
        training_copy_analysis(manifest['samples'],INDICES[split],directory,'copy',
                               reference=('All896 final-refit training songs, all bars' if split=='all'
                                          else 'All851 pre-refit optimization training songs, all bars;45 validation songs excluded'))
    else:
        boundaries=[];continuations=[]
        for record in manifest['samples']:
            detail=boundary_analysis(record['tokens'])
            ratio=detail['note_density_ratio'];jump=detail['pitch_mean_jump']
            detail['boundary_selection_score']=abs(math.log(max(ratio,1e-6)))+(abs(jump)/12 if jump is not None else 2)+(1-detail['pitch_class_jaccard'])
            detail['song_name']=record['song_name'];detail['configuration_id']=config
            detail['generated24bar_diagnostics']=piece_diagnostics(parse_bars(record['tokens'])[8:])
            boundaries.append(detail)
            offsets=[i for i,t in enumerate(record['tokens']) if t=='Bar']
            continuations.append({'song_name':record['song_name'],'tokens':['<BOS>']+record['tokens'][offsets[8]:]})
        write(directory/'boundary_diagnostics.json',boundaries)
        evaluate_basic_records(continuations,directory,3,24,config+'_continuation24')
    done[key]=digest;errors.pop(key,None)
    print(json.dumps({'evaluated':str(path.relative_to(ROOT)),'n':count,'H4':summary['summary']['H4']['mean'],'GS':summary['summary']['GS']['mean']}),flush=True)


def evaluate_bonus_manifest(path,tag,directory=None):
    manifest=read(path)
    if not manifest.get('complete') or len(manifest.get('samples',[]))!=10:return
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if done.get(str(path))==digest:return
    conditioning=manifest.get('conditioning',manifest['configuration'].get('conditioning','normal'))
    if tag=='bonus' and conditioning!='normal':raise ValueError('Formal bonus must retain normal melody conditioning')
    if tag=='bonus_ablation_no_melody' and (conditioning!='empty-melody' or manifest.get('submission_eligible') is not False):
        raise ValueError('Empty-melody ablation must be explicitly marked non-submission')
    if sorted(r['song_name'] for r in manifest['samples'])!=[f'{i:03d}' for i in range(1,11)]:
        raise ValueError('Bonus manifest IDs differ from001..010')
    for record in manifest['samples']:
        midi=path.parent/f'{record["song_name"]}.mid'
        if hashlib.sha256(midi.read_bytes()).hexdigest()!=record['midi_sha256']:
            raise ValueError('Bonus MIDI hash differs from generation manifest')
        if len(parse_bars(record['tokens']))!=16 or record['bars']!=16:
            raise ValueError('Bonus manifest token/bar count differs from16')
    directory=OUT/tag if directory is None else directory
    summary=evaluate_bonus_dir(path.parent,ROOT/'POP909/prompts/bonus',directory,tag=tag)
    keys=('task','checkpoint','checkpoint_sha256','configuration','conditioning_policy','all_official_training_songs_used','checkpoint_seen_training_songs','conditioning','submission_eligible','interpretation_limitations')
    summary['generation_provenance']={k:manifest[k] for k in keys if k in manifest}
    summary['generation_provenance']['manifest_sha256']=digest
    write(directory/f'{tag}_summary.json',summary)
    done[str(path)]=digest;errors.pop(str(path),None)
    print(json.dumps({'evaluated':tag,'n':10,'pooled':summary['pooled']}),flush=True)


def compare_bonus_conditioning():
    normal_path=OUT/'bonus_t10_reference/manifest.json';empty_path=BASE/'bonus_ablation_no_melody/manifest.json'
    if not all(str(p) in done for p in (normal_path,empty_path)):return
    normal,empty=read(normal_path),read(empty_path)
    if normal['checkpoint_sha256']!=empty['checkpoint_sha256']:
        raise ValueError('Conditioning ablation changed the roles checkpoint')
    for key in ('seed','temperature','top_k','top_p'):
        if normal['configuration'][key]!=empty['configuration'][key]:
            raise ValueError(f'Conditioning ablation changed sampling setting {key}')
    config=normal['configuration']
    if config['temperature']!=1. or config['top_p']!=.95 or config['top_k']!=0:
        raise ValueError('Conditioning ablation must use the predefined T1/top-p.95/top-k0 settings')
    empty_by_id={r['song_name']:r for r in empty['samples']}
    for record in normal['samples']:
        other=empty_by_id[record['song_name']]
        for key in ('input_melody_sha256','input_beats_sha256'):
            if record[key]!=other[key]:raise ValueError(f'Conditioning ablation changed original input {key}')
        if record['diagnostics']['seed']!=other['diagnostics']['seed']:
            raise ValueError('Conditioning ablation prompt seeds differ')
    normal_rows=pd.read_csv(OUT/'bonus_t10_reference/bonus_t10_reference_per_piece.csv',dtype={'song_name':str}).to_dict('records')
    empty_rows=pd.read_csv(OUT/'bonus_ablation_no_melody/bonus_ablation_no_melody_per_piece.csv',dtype={'song_name':str}).to_dict('records')
    aliases={'CA':'CA','OA_P':BONUS.OA_COLS['DP'],'OA_V':BONUS.OA_COLS['DV'],'OA_D':BONUS.OA_COLS['DD'],'OA_IOI':BONUS.OA_COLS['DIOI']}
    paired=paired_metric_comparison(empty_rows,normal_rows,metrics=tuple(aliases.values()),seed=6221)
    normal_by_id={r['song_name']:r for r in normal_rows};empty_by_id={r['song_name']:r for r in empty_rows}
    records=[];table=[];metrics={}
    for song in sorted(normal_by_id):
        n,e=normal_by_id[song],empty_by_id[song]
        if n['n_bars']!=16 or e['n_bars']!=16 or abs(n['CA-ceiling']-e['CA-ceiling'])>1e-12:
            raise ValueError('Conditioning ablation reference bars or CA-ceiling changed')
        row={'song_name':song}
        for short,long in aliases.items():row.update({f'normal_{short}':n[long],f'empty_{short}':e[long],f'normal_minus_empty_{short}':n[long]-e[long]})
        records.append(row)
    pd.DataFrame(records).to_csv(OUT/'bonus_conditioning_ablation_per_prompt.csv',index=False)
    for short,long in aliases.items():
        d=paired['metrics'][long];n=sum(r[long] for r in normal_rows)/10;e=sum(r[long] for r in empty_rows)/10
        metrics[short]={'official_name':long,'normal_mean':n,'empty_mean':e,**d}
        table.append([short,n,e,d['mean_second_minus_first'],*d['paired_mean_ci95']])
    result={'n_pairs':10,'expected_pairs':10,'skipped_pairs':0,'difference_direction':'normal minus empty-melody','metrics':metrics,
            'method':paired['method'],'checkpoint_sha256':normal['checkpoint_sha256'],'configuration':config,
            'normal_reference':'Frozen original T1 normal-melody generation, retained independently of any later formal bonus selection.',
            'normal_manifest_sha256':done[str(normal_path)],'ablation_manifest_sha256':done[str(empty_path)],
            'melody_and_beat_inputs_identical':True,'original_output_melody_preservation_checked':True,
            'submission_eligible':False,'table_headers':['Metric','Normal','Empty melody','Normal-empty','95% low','95% high'],'table_rows':table,
            'interpretation':'The sign of each measured normal-minus-empty difference is reported without assuming the normal condition wins. Entire prompts, not bars, are paired and resampled.',
            'limitations':'Removing all MELODY notes is an out-of-distribution intervention and changes context length and later sampled histories. It cannot isolate a causal effect on musical quality. Both outputs restore exactly the original melody; the empty variant is evaluation-only and is excluded from the formal submission.'}
    positive=[k for k,v in metrics.items() if v['paired_mean_ci95'][0]>0]
    negative=[k for k,v in metrics.items() if v['paired_mean_ci95'][1]<0]
    crossing=[k for k,v in metrics.items() if v['paired_mean_ci95'][0]<=0<=v['paired_mean_ci95'][1]]
    result['interval_excludes_zero_positive']=positive;result['interval_excludes_zero_negative']=negative
    result['interval_includes_zero']=crossing
    result['interpretation']=(f"Across these ten paired prompts, normal-minus-empty intervals lie above zero for {', '.join(positive) or 'no metrics'}, below zero for {', '.join(negative) or 'no metrics'}, and include zero for {', '.join(crossing) or 'no metrics'}. The five percentile intervals are unadjusted; they describe this intervention and do not establish a causal improvement in musical quality.")
    write(OUT/'bonus_conditioning_ablation.json',result)


def compare_bonus_sampling():
    specs={'t08':(BASE/'bonus_sampling/t08/manifest.json',OUT/'bonus_sampling/t08/t08_per_piece.csv'),
           't10':(OUT/'bonus_t10_reference/manifest.json',OUT/'bonus_t10_reference/bonus_t10_reference_per_piece.csv'),
           't12':(BASE/'bonus_sampling/t12/manifest.json',OUT/'bonus_sampling/t12/t12_per_piece.csv')}
    if not all(str(path) in done and csv.exists() for path,csv in specs.values()):return
    manifests={c:read(path) for c,(path,_) in specs.items()}
    if len({m['checkpoint_sha256'] for m in manifests.values()})!=1:raise ValueError('Bonus sampling comparison changed checkpoint')
    if len({m['configuration']['seed'] for m in manifests.values()})!=1:raise ValueError('Bonus sampling comparison changed seed schedule')
    aliases={'CA':'CA','OA_P':BONUS.OA_COLS['DP'],'OA_V':BONUS.OA_COLS['DV'],'OA_D':BONUS.OA_COLS['DD'],'OA_IOI':BONUS.OA_COLS['DIOI']}
    records={};summary={};rows=[];flat=[]
    draws=np.random.default_rng(6221).integers(0,10,(5000,10))
    reference={r['song_name']:r for r in manifests['t10']['samples']}
    for c,(path,csv) in specs.items():
        m=manifests[c];config=m['configuration'];expected_temperature={'t08':.8,'t10':1.,'t12':1.2}[c]
        if config['temperature']!=expected_temperature or config['top_p']!=.95 or config['top_k']!=0:
            raise ValueError('Bonus sampling settings differ from the predefined temperature comparison')
        if m.get('conditioning',config.get('conditioning','normal'))!='normal':raise ValueError('Bonus sampling comparison changed melody conditioning')
        for r in m['samples']:
            for key in ('input_melody_sha256','input_beats_sha256'):
                if r[key]!=reference[r['song_name']][key]:raise ValueError('Bonus sampling input hashes differ')
            if r['diagnostics']['seed']!=reference[r['song_name']]['diagnostics']['seed']:raise ValueError('Bonus sampling per-prompt seeds differ')
        data=pd.read_csv(csv,dtype={'song_name':str}).to_dict('records')
        if len(data)!=10 or any(r['n_bars']!=16 for r in data):raise ValueError('Bonus sampling comparison lost required prompts/bars')
        records[c]=[]
        for r in data:
            d={'song_name':r['song_name'],**{short:r[long] for short,long in aliases.items()}}
            d['composite']=sum(d[k] for k in aliases)/5;records[c].append(d);flat.append({'configuration':c,**d})
        means={k:float(np.mean([r[k] for r in records[c]])) for k in (*aliases,'composite')}
        values=np.array([r['composite'] for r in records[c]])
        summary[c]={'temperature':expected_temperature,'n_prompts':10,'mean_metrics':means,
                    'composite_mean_ci95':np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                    'source_manifest_sha256':done[str(path)]}
        rows.append([expected_temperature,*[means[k] for k in aliases],means['composite']])
    paired={c:paired_metric_comparison(records['t10'],records[c],metrics=(*aliases,'composite'),seed=6221) for c in ('t08','t12')}
    selected=min(summary,key=lambda c:(-summary[c]['mean_metrics']['composite'],c))
    runner=sorted(summary,key=lambda c:(-summary[c]['mean_metrics']['composite'],c))[1]
    selected_difference=paired_metric_comparison(records[runner],records[selected],metrics=('composite',),seed=6221)['metrics']['composite']
    pd.DataFrame(flat).to_csv(OUT/'bonus_sampling_comparison_per_prompt.csv',index=False)
    result={'selected_configuration':selected,'selected_temperature':summary[selected]['temperature'],'conditions':summary,
            'selection_rule':'Maximize equal-weight mean(CA,OA_P,OA_V,OA_D,OA_IOI); lexical configuration ID breaks exact ties.',
            'n_prompts':10,'skipped_prompts':0,'selection_uses_same_reported_prompts':True,'independent_test':False,
            'runner_up_configuration':runner,'selected_minus_runner_up_composite':selected_difference,
            'paired_differences_vs_t10':paired,'bootstrap':'5000 paired whole-prompt draws with seed6221; percentile95 intervals.',
            'table_headers':['T','CA','OA_P','OA_V','OA_D','OA_IOI','Mean'],'table_rows':rows,
            'replacement_performed_by_this_analysis':False,
            'limitation':'Temperature selection uses the same10 provided prompts and evaluation ground truth; this is in-sample tuning without an independent test. A larger composite or individual metric does not establish better music or listener preference. The fixed T1 conditioning-ablation baseline is preserved separately.'}
    write(OUT/'bonus_sampling_comparison.json',result)


def compare_real_references():
    full_path=OUT/'real_reference_all/real_reference_summary.json'
    if not full_path.exists():return
    full=read(full_path);primary_selection=read(OUT/'real_reference/real_reference_selection.json')
    full_selection=read(OUT/'real_reference_all/real_reference_selection.json')
    first=pd.read_csv(OUT/'real_reference/real_reference_per_piece.csv',dtype={'song_name':str})
    second=pd.read_csv(OUT/'real_reference_all/real_reference_per_piece.csv',dtype={'song_name':str})
    metric_names=('H4','GS','notes_per_bar');differences={}
    for metric in metric_names:
        a,b=REF['summary'][metric],full['summary'][metric]
        grid=np.unique(np.concatenate([first[metric],second[metric]]))
        cdf_a=np.searchsorted(np.sort(first[metric]),grid,side='right')/len(first)
        cdf_b=np.searchsorted(np.sort(second[metric]),grid,side='right')/len(second)
        differences[metric]={'primary45_mean':a['mean'],'all_mean':b['mean'],
                             'primary_minus_all':a['mean']-b['mean'],
                             'standardized_primary_minus_all':(a['mean']-b['mean'])/max(b['std'],1e-6),
                             'all_mean_inside_primary45_mean_ci95':a['mean_ci95'][0]<=b['mean']<=a['mean_ci95'][1],
                             'descriptive_maximum_cdf_distance':float(np.max(np.abs(cdf_a-cdf_b)))}
    result={'primary_reference_unchanged':True,'basic_a_selection_rule_unchanged':True,
            'primary_included':primary_selection['included'],'primary_excluded':len(primary_selection['excluded']),
            'all_included':full_selection['included'],'all_excluded':len(full_selection['excluded']),
            'primary_song_ids_subset_of_all':set(first.song_name)<=set(second.song_name),
            'metrics':differences,'full_exclusions':full_selection['excluded'],
            'method':'Same deterministic center-nearest eligible32bar crop per song and identical canonical16subdivision/128position adapter. All896 official songs were considered; short or unrepresentable songs are listed individually.',
            'limitation':'The45song set is contained in the full set; these comparisons are descriptive and are not independent hypothesis tests. Agreement on H4/GS/density does not establish representative musical style or perceived quality.'}
    covered=sum(v['all_mean_inside_primary45_mean_ci95'] for v in differences.values())
    shifts=', '.join(f"{k}={v['standardized_primary_minus_all']:+.3f} full-set standard deviations" for k,v in differences.items())
    result['assessment']=f"Primary45 minus all-reference mean shifts are {shifts}. The all-reference mean falls inside the primary45 whole-piece bootstrap interval for {covered}/3 descriptors. The subset closely matches the full eligible set under these measured descriptors; this supports its descriptive use here, without establishing representative musical style or perceived quality."
    selection_path=OUT/'selection_progress.json'
    if selection_path.exists():
        selected=read(selection_path)['selected_configuration'];generated=read(OUT/f'basic_a/{selected}/{selected}_summary.json')
        gaps={}
        for label,reference in [('primary45',REF),('all_eligible',full)]:
            gaps[label]={'selection_metric_distance':compare_to_reference(generated,reference),
                         'generated_minus_reference':{k:generated['summary'][k]['mean']-reference['summary'][k]['mean'] for k in metric_names}}
        result['selected_configuration']=selected;result['selected_configuration_reference_gaps']=gaps
        rankings={label:{c:compare_to_reference(read(OUT/f'basic_a/{c}/{c}_summary.json'),reference)['mean_gap'] for c in A}
                  for label,reference in [('primary45',REF),('all_eligible',full)]}
        result['hypothetical_configuration_scores']=rankings
        result['minimum_score_configuration']={label:min(scores,key=lambda c:(scores[c],c)) for label,scores in rankings.items()}
    write(OUT/'reference_sensitivity.json',result)


def aggregate():
    summaries={c:read(OUT/f'basic_a/{c}/{c}_summary.json') for c in A if (OUT/f'basic_a/{c}/{c}_summary.json').exists()}
    if len(summaries)!=len(A):return False
    hashes={d['generation_provenance']['checkpoint_sha256'] for d in summaries.values()}
    seeds={d['generation_provenance']['configuration']['seed'] for d in summaries.values()}
    if len(hashes)!=1 or len(seeds)!=1:raise ValueError('Basic-a settings must share one checkpoint and one seed schedule')
    comparisons={c:compare_to_reference(d,REF) for c,d in summaries.items()}
    selected=min(A,key=lambda c:(comparisons[c]['mean_gap'],c))
    progress={'selected_configuration':selected,'selection_reasoning':'Minimum equal-weight mean standardized absolute H4/GS gap to the45song pre-refit real-data reference; objective symbolic selection, no human listening.',
              'configuration_comparisons':comparisons,'human_listening_performed':False,
              'configuration_manifest_sha256':{c:d['generation_provenance']['manifest_sha256'] for c,d in summaries.items()},
              'reference_scope':REF['reference_scope'],'selection_on_reported_samples':True}
    write(OUT/'selection_progress.json',progress)
    if not all((OUT/f'basic_b/{c}/boundary_diagnostics.json').exists() for c in B):return False
    for c in B:
        provenance=read(OUT/f'basic_b/{c}/{c}_summary.json')['generation_provenance']
        if provenance['checkpoint_sha256'] not in hashes:raise ValueError('Basic-b checkpoint differs from basic-a checkpoint')
    boundaries=[r for c in B for r in read(OUT/f'basic_b/{c}/boundary_diagnostics.json')]
    best={p:min((r for r in boundaries if r['song_name']==p),key=lambda r:(r['boundary_selection_score'],r['configuration_id']))['configuration_id'] for p in ('001','002','003')}
    prompts=[]
    for p,c in best.items():
        rows=[r for r in boundaries if r['song_name']==p]
        winner=next(r for r in rows if r['configuration_id']==c)
        jump=('unavailable because a window is silent' if winner['pitch_mean_jump'] is None else f"{winner['pitch_mean_jump']:.3f} semitones")
        prompts.append({'prompt_id':p,'best_configuration':c,'analysis':f"{c} minimizes the specified two-bar boundary score ({winner['boundary_selection_score']:.4f}); density ratio={winner['note_density_ratio']:.3f}, pitch mean jump={jump}, pitch-class Jaccard={winner['pitch_class_jaccard']:.3f}.",
                        'discussion':'This is a reproducible local continuity preference, with all original prefix notes verified. It is not a listener judgment or evidence of long-term musical development.',
                        'comparison':rows})
    allrows={c:pd.read_csv(OUT/f'basic_a/{c}/{c}_per_piece.csv').to_dict('records') for c in A}
    paired={c:paired_metric_comparison(allrows['t10'],allrows[c]) for c in A if c!='t10'}
    write(OUT/'paired_sampling_comparisons.json',paired)
    headers=['Set','n','H4','GS','GS_active','Notes/bar','Empty bars']
    def metric_row(name,d):return [name,d['evaluated_pieces']]+[d['summary'][m]['mean'] for m in ('H4','GS','GS_active','notes_per_bar','empty_bar_fraction')]
    sampling_rows=[metric_row(c,summaries[c]) for c in A]
    checkpoint_rows=[]
    for path in sorted(OUT.glob('checkpoint*/*_summary.json')):
        d=read(path)
        if d.get('evaluated_pieces')==20 and 'H4' in d.get('summary',{}):checkpoint_rows.append(metric_row(path.parent.name,d))
    copies={c:read(OUT/f'basic_a/{c}/copy_summary.json') for c in A}
    copy_rows=[[c,copies[c]['duplicate_full_piece_count']]+[copies[c]['mean_match_fractions'][f'exact_{w}bar_nonempty_match_fraction'] for w in (1,2,4,8)] for c in A]
    boundary_rows=[[r['song_name'],r['configuration_id'],r['boundary_selection_score'],r['note_density_ratio'],r['pitch_mean_jump'],r['pitch_class_jaccard']] for r in boundaries]
    experiments=[
      {'name':'Sampling temperature and truncation','method':'Frozen final checkpoint,20 independent32bar samples per setting, identical seed schedule and canonical official metric adapter. Whole-piece bootstrap confidence intervals are stored with the primary metrics.',
       'headers':headers,'rows':sampling_rows,'analysis':f"The stated metric-distance rule selects {selected}, with standardized mean gap={comparisons[selected]['mean_gap']:.4f}. All settings retain their complete20piece output sets.",
       'discussion':'Entropy describes pitch-class diversity; GS describes onset-pattern consistency. Density, rest and repetition diagnostics are needed to interpret them. Paired5000draw whole-piece bootstrap differences versus t10 are saved in paired_sampling_comparisons.json.',
       'limitations':'The same reported sets are used for configuration selection. The real-data reference is not independent of the all896song final refit. No human listening study was performed.'},
      {'name':'Checkpoint learning diagnostics','method':'Diagnostic checkpoints are evaluated on independently generated20piece32bar sets using the same seed schedule and canonical adapter.',
       'headers':headers,'rows':checkpoint_rows,'analysis':'Only completed20piece sets are included; partial generation manifests are excluded from this comparison.',
       'discussion':'The500step model already has GS close to the real reference while generating roughly half its note density. Similar GS therefore does not establish realistic arrangement or musical quality.',
       'limitations':'These are pre-refit diagnostic checkpoints, not the all896song final model. Different effective contexts can also influence generation, so checkpoint changes are not an isolated causal experiment.'},
      {'name':'Official GS normalization sensitivity','method':'Keep the unchanged official128position GS and also report a separately named diagnostic using max(meter_a,meter_b)*8 active positions per bar pair.',
       'headers':headers,'rows':[metric_row('Pre-refit real reference',REF)]+sampling_rows,
       'analysis':'The canonical adapter maps the model8subdivisions/beat to16 by multiplying positions by2. Odd positions and positions outside the active meter remain zero; this raises official GS relative to the active-grid diagnostic.',
       'discussion':'Both scores are shown without substituting the diagnostic for the required official metric. Real and generated sets use exactly the same adapter.',
       'limitations':'GS ignores pitch, duration, note velocity and note multiplicity at one onset. It cannot certify melody, harmony or expressive phrasing.'},
      {'name':'Exact local training matches','method':'Hash contiguous bars as position8/pitch/duration8 multisets, ignoring tempo, velocity and track names; compare every generated window to every bar of all896 final-refit training songs.',
       'headers':['Configuration','Duplicate pieces','Nonempty1bar','Nonempty2bar','Nonempty4bar','Nonempty8bar'],'rows':copy_rows,
       'analysis':'Nonempty-window match fractions exclude fully silent windows. Raw silent-window matches remain available in each copy_per_piece.csv, avoiding an interpretation of shared silence as copied musical phrases.',
       'discussion':'Every generated piece is retained in this check, with no metric-based removal.',
       'limitations':'Zero exact matches does not establish originality: transposed, approximate, shifted, velocity-only or partial-note copying can escape this test.'},
      {'name':'Continuation boundary continuity','method':'Compare two bars immediately before and after the8bar prompt boundary. Within each prompt minimize |log density ratio|+|pitch mean jump|/12+(1-pitch-class Jaccard).',
       'headers':['Prompt','Selected setting','Score','Density ratio','Pitch mean jump','PC Jaccard'],'rows':[r for r in boundary_rows if best[r[0]]==r[1]],
       'analysis':'Best settings under this fixed symbolic rule are '+', '.join(f'{p}:{c}' for p,c in best.items())+'. Original8bar note pitch, velocity and timing identity are separately checked from the MIDI files.',
       'discussion':'A low boundary score indicates a smaller local change in note density and pitch content; abrupt changes can still be musically valid.',
       'limitations':'Two-bar statistics do not assess phrase completion, later24bar development or listener preference. Supplemental full32bar H4/GS includes the supplied prefix; continuation-only24bar metrics are saved separately.'}
    ]
    context_path=BASE/'context_comparison.json'
    if context_path.exists():
        context=read(context_path);branches=context['branches']
        if len({b['parent_sha256'] for b in branches})!=1:raise ValueError('Context comparison branches have different parents')
        rows=[[b['context'],b['processed_tokens'],b['validation_loss']] for b in branches]
        experiments.append({'name':'Context length at a matched token budget',
                            'method':'Continue two branches from the same saved parent checkpoint. Compare their last checkpoints after approximately four million non-padding targets; the budget can overshoot by at most one optimizer step. Intermediate best checkpoints are excluded.',
                            'headers':['Context tokens','Processed targets','Validation NLL'],'rows':rows,
                            'analysis':f"The saved comparison selects context {context['selected']['context']} with validation cross-entropy {context['selected']['validation_loss']:.4f} nats. The 2,048-token branch scores {next(b['validation_loss'] for b in branches if b['context']==2048):.4f}, compared with {next(b['validation_loss'] for b in branches if b['context']==4096):.4f} for 4,096 tokens.",
                            'discussion':'A longer context did not improve this continuation experiment under its measured budget. The selected branch is followed by the all-song refit.',
                            'limitations':'There is one branch per setting. Different sequence lengths change training windows and validation windows, and the token budgets are approximate. This does not establish that shorter context is generally better or isolate musical quality.',
                            'evidence_files':['results/context_comparison.json','artifacts/models/context_2048/history.json','artifacts/models/context_4096/history.json']})
    token_paths=[BASE/f'token_diagnostics_{step:04d}.json' for step in (500,2000)]
    if all(p.exists() for p in token_paths):
        early,later=[read(p) for p in token_paths]
        for key in ('windows_sha256','validation_song_ids','observed_song_ids','batch_target_counts'):
            if early['evaluation'][key]!=later['evaluation'][key]:raise ValueError('Token diagnostics do not share fixed evaluation windows')
        if early['overall']['targets']!=later['overall']['targets'] or not all(d['evaluation']['held_out_from_checkpoint_training'] for d in (early,later)):
            raise ValueError('Token diagnostic counts or pre-refit held-out scope differ')
        families=('Pos','Pitch','Vel','Dur','Tempo','Meter')
        rows=[[name,early['by_target_type'][name]['targets'],early['by_target_type'][name]['cross_entropy_nats'],later['by_target_type'][name]['cross_entropy_nats'],early['by_target_type'][name]['accuracy'],later['by_target_type'][name]['accuracy']] for name in families]
        experiments.append({'name':'Next-token errors by musical event family',
                            'method':f"Evaluate both diagnostic checkpoints on the same {early['evaluation']['windows']} deterministic windows from all {early['evaluation']['observed_songs']} pre-refit validation songs: {early['overall']['targets']:,} non-padding targets. Use full-vocabulary next-token scores, without grammar or sampling masks.",
                            'headers':['Family','Targets','NLL 500','NLL 2000','Accuracy 500','Accuracy 2000'],'rows':rows,
                            'analysis':f"Overall cross-entropy decreases from {early['overall']['cross_entropy_nats']:.4f} to {later['overall']['cross_entropy_nats']:.4f} nats; full-vocabulary top-1 accuracy rises from {early['overall']['accuracy']:.2%} to {later['overall']['accuracy']:.2%}. Pitch remains difficult ({later['by_target_type']['Pitch']['cross_entropy_nats']:.4f} nats), compared with position ({later['by_target_type']['Pos']['cross_entropy_nats']:.4f}).",
                            'discussion':'The matched window hash and target counts prevent a changing evaluation sample from explaining the checkpoint difference. Six event families are shown; the saved source also retains Bar and EOS statistics.',
                            'limitations':'These are pre-refit teacher-forced diagnostics. Better token prediction does not establish better long-form music. Type-conditioned scores in the source know the target family and must not be treated as unconstrained prediction accuracy.',
                            'evidence_files':['results/token_diagnostics_0500.json','results/token_diagnostics_2000.json']})
    for experiment in experiments:
        experiment.setdefault('evidence_files',[str(p.relative_to(ROOT)) for p in sorted(OUT.rglob('*.json'))
                                      if not any(part=='cache' for part in p.parts)
                                      and 'sequences' not in p.stem and p.name.endswith(('_summary.json','boundary_diagnostics.json','paired_sampling_comparisons.json','official_cli_agreement.json','environment_adapter_audit.json'))])
    ablation_path=OUT/'bonus_conditioning_ablation.json'
    if ablation_path.exists():
        ablation=read(ablation_path)
        experiments.append({'name':'Bonus melody-conditioning ablation','method':'Same roles checkpoint,10 identical prompts and seed schedule,T1/top-p.95. Remove MELODY note quartets while keeping bar/meter/tempo/track headers; restore the complete original melody in both exports. Evaluate the unchanged official CA/OA and bootstrap paired whole prompts5000times.',
                            'headers':ablation['table_headers'],'rows':ablation['table_rows'],'analysis':ablation['interpretation'],
                            'discussion':'Positive normal-minus-empty differences favor the normal condition on that metric; negative differences favor the empty condition. Confidence intervals and all10 paired prompt rows are retained.',
                            'limitations':ablation['limitations'],
                            'evidence_files':['results/evaluation/bonus_conditioning_ablation.json','results/evaluation/bonus_conditioning_ablation_per_prompt.csv','results/evaluation/bonus_t10_reference/bonus_t10_reference_summary.json','results/evaluation/bonus_t10_reference/bonus_t10_reference_per_piece.csv','results/evaluation/bonus_ablation_no_melody/bonus_ablation_no_melody_summary.json','results/evaluation/bonus_ablation_no_melody/bonus_ablation_no_melody_per_piece.csv']})
    sampling_path=OUT/'bonus_sampling_comparison.json'
    if sampling_path.exists():
        sampling=read(sampling_path);winner=sampling['selected_configuration'];condition=sampling['conditions'][winner]
        difference=sampling['selected_minus_runner_up_composite'];low,high=difference['paired_mean_ci95']
        experiments.append({'name':'Bonus sampling temperature selection',
                            'method':'Hold the roles checkpoint, ten melody prompts, seed schedule and top-p .95 fixed; compare normal-conditioning temperatures .8, 1.0 and 1.2. Select the largest equal-weight average of the five official CA/OA metrics.',
                            'headers':sampling['table_headers'],'rows':sampling['table_rows'],
                            'analysis':f"The predefined composite selects {winner} (T={sampling['selected_temperature']:.1f}), mean={condition['mean_metrics']['composite']:.4f}. Its difference from runner-up {sampling['runner_up_configuration']} is {difference['mean_second_minus_first']:+.4f}, paired 95% interval [{low:+.4f}, {high:+.4f}]. The fixed T1 baseline is retained for the conditioning ablation.",
                            'discussion':'All ten prompts remain in each condition. Paired prompt-bootstrap differences from T1 and all per-prompt composite values are saved with the evidence.',
                            'limitations':sampling['limitation'],
                            'evidence_files':['results/evaluation/bonus_sampling_comparison.json','results/evaluation/bonus_sampling_comparison_per_prompt.csv','results/evaluation/bonus_sampling/t08/t08_summary.json','results/evaluation/bonus_sampling/t08/t08_per_piece.csv','results/evaluation/bonus_sampling/t12/t12_summary.json','results/evaluation/bonus_sampling/t12/t12_per_piece.csv','results/evaluation/bonus_t10_reference/bonus_t10_reference_summary.json','results/evaluation/bonus_t10_reference/bonus_t10_reference_per_piece.csv']})
    sensitivity_path=OUT/'reference_sensitivity.json'
    if sensitivity_path.exists():
        sensitivity=read(sensitivity_path);full=read(OUT/'real_reference_all/real_reference_summary.json')
        rows=[[label,n,excluded]+[reference['summary'][k]['mean'] for k in ('H4','GS','notes_per_bar')]
              for label,n,excluded,reference in [('Primary45',sensitivity['primary_included'],sensitivity['primary_excluded'],REF),
                                                 ('All eligible',sensitivity['all_included'],sensitivity['all_excluded'],full)]]
        gaps=sensitivity.get('selected_configuration_reference_gaps',{})
        comparison=' '.join(f"Selected {selected} minus {label}: "+', '.join(f'{k}={v:+.4f}' for k,v in detail['generated_minus_reference'].items())+'.' for label,detail in gaps.items())
        experiments.append({'name':'Real-data reference sensitivity','method':sensitivity['method'],
                            'headers':['Reference','Included','Excluded','H4','GS','Notes/bar'],'rows':rows,
                            'analysis':sensitivity['assessment'],'discussion':comparison+' The primary45 reference and selected configuration are retained exactly as specified.',
                            'limitations':sensitivity['limitation'],
                            'evidence_files':['results/evaluation/reference_sensitivity.json','results/evaluation/real_reference_all/real_reference_summary.json','results/evaluation/real_reference_all/real_reference_per_piece.csv','results/evaluation/real_reference_all/real_reference_selection.json']})
    reasons={p:{r['configuration_id']:f"Boundary score={r['boundary_selection_score']:.4f}; density ratio={r['note_density_ratio']:.3f}; pitch mean jump={r['pitch_mean_jump']}; pitch-class Jaccard={r['pitch_class_jaccard']:.3f}." for r in boundaries if r['song_name']==p} for p in best}
    limitations=['No human listening or preference ratings were collected; selections use the stated symbolic proxies.',
                 'The primary 45-song reference was held out only before the final fit. Both the 45-song and 891-song real references overlap the final model trained on all 896 songs; the final model has no independent test set.',
                 'Sampling settings are selected using the displayed generated sets. Bootstrap intervals describe variability over pieces or prompts for one checkpoint, without repeated training seeds.',
                 'Generation uses grammar masks, a 64-note cap per bar, fixed generated meter/tempo and context rollover. These choices constrain expression and long-range memory.',
                 'Eight positions per beat, quantized velocity and a 32-beat duration cap lose source detail. Original continuation prefixes and bonus melodies are restored after generation and verified separately.',
                 'H4 measures pitch-class diversity and GS measures onset consistency with padded position normalization. Neither certifies musical quality; OA histograms also omit note order and pitch register.',
                 'The bonus model conditions on melody and beat timing, without explicit chord input. Its measured chord agreement remains low relative to the ground-truth piano self-reference.',
                 'Zero exact training-window matches does not rule out approximate, transposed or partial-note copying.',
                 'The official metrics are descriptive proxies; no numeric formula converts these values into a course grade.']
    if ablation_path.exists():limitations.append(read(ablation_path)['limitations'])
    if sampling_path.exists():limitations.append(read(sampling_path)['limitation'])
    final_audit=OUT/'final_integrity_audit.json'
    evidence_files=([str(final_audit.relative_to(ROOT))] if final_audit.exists() else [])
    result={**progress,'basic_a_selection':{'method':progress['selection_reasoning'],'scores':comparisons},
            'basic_b_best':best,'basic_b_reasons':reasons,'basic_b_prompt_analysis':{p['prompt_id']:p['analysis']+' '+p['discussion'] for p in prompts},
            'prompt_analysis':prompts,'boundary_summaries':boundaries,
            'additional_experiments':experiments,'failures':[],
            'limitations':limitations,'evidence_files':evidence_files,
            'bonus_method':'The roles Transformer reads only the supplied melody and annotation beat grid, forces MELODY tokens as context at each bar, and samples PIANO notes. Export restores the original melody notes; ground-truth piano/chords are used only by the evaluator.'}
    write(OUT/'report_analysis.json',result)
    return True


if __name__=='__main__':
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        jobs=[(p,'checkpoint',p.parent.name,20) for p in BASE.glob('checkpoint*/manifest.json')]
        jobs += [(BASE/f'basic_a/{c}/manifest.json','basic_a',c,20) for c in A]
        jobs += [(BASE/f'basic_b/{c}/manifest.json','basic_b',c,3) for c in B]
        for path,kind,config,count in jobs:
            if not path.exists():continue
            try:evaluate_manifest(path,kind,config,count)
            except Exception:
                trace=traceback.format_exc()
                if errors.get(str(path))!=trace:print(trace,flush=True)
                errors[str(path)]=trace
        for tag in ('bonus','bonus_ablation_no_melody'):
            bonus_path=BASE/tag/'manifest.json'
            if not bonus_path.exists():continue
            try:
                evaluate_bonus_manifest(bonus_path,tag)
            except Exception:
                trace=traceback.format_exc()
                if errors.get(str(bonus_path))!=trace:print(trace,flush=True)
                errors[str(bonus_path)]=trace
        extra_bonus=[(OUT/'bonus_t10_reference/manifest.json','bonus_t10_reference',OUT/'bonus_t10_reference')]
        extra_bonus += [(BASE/f'bonus_sampling/{c}/manifest.json',c,OUT/f'bonus_sampling/{c}') for c in ('t08','t12')]
        for path,tag,directory in extra_bonus:
            if not path.exists():continue
            try:evaluate_bonus_manifest(path,tag,directory)
            except Exception:
                trace=traceback.format_exc()
                if errors.get(str(path))!=trace:print(trace,flush=True)
                errors[str(path)]=trace
        try:
            compare_bonus_conditioning()
            compare_bonus_sampling()
            compare_real_references()
            complete=aggregate();errors.pop('aggregation',None)
        except Exception:
            complete=False;errors['aggregation']=traceback.format_exc()
        audio_path=OUT/'finish_audio_status.json'
        final_paths=[BASE/f'basic_a/{c}/manifest.json' for c in A]+[BASE/f'basic_b/{c}/manifest.json' for c in B]+[BASE/f'{tag}/manifest.json' for tag in ('bonus','bonus_ablation_no_melody')]
        current=all(p.exists() and done.get(str(p))==hashlib.sha256(p.read_bytes()).hexdigest() for p in final_paths)
        audio_complete=audio_path.exists() and read(audio_path).get('complete') is True
        sampling_ready=(OUT/'bonus_sampling_comparison.json').exists()
        all_complete=complete and current and audio_complete and sampling_ready and not errors
        write(OUT/'watch_status.json',{'complete_manifests_evaluated':done,'errors':errors,'report_analysis_ready':complete,
                                      'normal_bonus_evaluated':str(BASE/'bonus/manifest.json') in done,
                                      'ablation_bonus_evaluated':str(BASE/'bonus_ablation_no_melody/manifest.json') in done,
                                      'reference_sensitivity_ready':(OUT/'reference_sensitivity.json').exists(),
                                      'bonus_sampling_comparison_ready':sampling_ready,
                                      'formal39_audio_complete':audio_complete,'all_final_evaluations_complete':all_complete})
        if all_complete:
            print('ALL_FINAL_EVALUATIONS_COMPLETE',flush=True);break
        time.sleep(20)
