"""Rebuild and audit the #95/#96 report from committed result artifacts.

Run: python -m squadqa.report_answer_distribution
No checkpoint, GPU, cluster connection or external dataset download is needed.
"""
import hashlib,json
from pathlib import Path
from squadqa.answer_distribution import summary, ROW1, ROW0
import numpy as np
def main():
    stage=Path('results')
    all_data={}
    for run in ROW1+ROW0:
     for prefix, mode, count, job in [('distribution_validation','shape',7708,'1048461')]+([('multi_answer','multi-answer',1250,'1048460')] if run in ROW1 else []):
      sf=stage/run/(prefix+'.json');pf=stage/run/(prefix+'_questions.json')
      s=json.loads(sf.read_text());p=json.loads(pf.read_text());m=s['metadata']
      assert m==p['metadata'] and m['questions']==count and m['slurm_job_id']==job
      assert m['analysis_commit']=='aaba686bc8ac7980a1052415928aaeb6063fdc13' and not m['git_dirty'] and m['limit'] is None
      assert hashlib.sha256((Path('results')/run/'config.yaml').read_bytes()).hexdigest()==m['reader_config_sha256']
      assert hashlib.sha256((Path('results')/run/'calibration.json').read_bytes()).hexdigest()==m['calibration_sha256']
      for label in ['at_1','calibrated']:
       assert len(p[label])==count and s[label]==summary(p[label],mode)
       if mode=='shape':
        for r in p[label].values():
         assert r['group_entropy']<=r['span_entropy']+1e-8
         if r['mass_overlapping_gold'] is not None: assert abs(r['mass_overlapping_gold']+r['mass_disjoint_gold']+r['null_mass']-1)<1e-8
       else:
        assert sum(r['split']=='heldout' for r in p[label].values())==250
        assert sum(r['eligible_2_regions'] for r in p[label].values())==1247
        for r in p[label].values():
         assert abs(sum(r['q_regions'])-1)<1e-9
         assert abs(sum(r['p_regions'])+r['outside_mass']-1)<1e-8
      all_data[(run,prefix)]=(s,p)
    print('18 JSON artifacts validated; all per-question summaries recomputed exactly.')

    def ms(xs):return {'mean':float(np.mean(xs)), 'sample_std':float(np.std(xs,ddof=1)), 'seeds':list(map(float,xs))}
    report={'analysis_commit':'aaba686bc8ac7980a1052415928aaeb6063fdc13','jobs':{'setup':1048451,'smoke':1048453,'multi_answer':1048460,'shape':1048461},'tests':{'passed':389,'skipped':13},'nq':{},'shape':{},'validation_checks':{}}
    for t in ['at_1','calibrated']:
     report['nq'][t]={}
     for split in ['all','heldout']:
      ds=[all_data[(r,'multi_answer')][0][t][split] for r in ROW1]
      out={'total':ds[0]['total'],'eligible':ds[0]['eligible']}
      for k in ['answer_region_mass','collapse','q_top_region_share','second_to_first_mass','second_region_in_top5','second_region_in_top10','kl_q_p','js_distance','top_region_is_first_annotated']:
       out[k]=ms([d['measurements'][k]['mean'] for d in ds])
      for cutoff in ['0.9','0.99']:out['collapse_above_'+cutoff]=ms([d['measurements']['collapse_shares'][cutoff] for d in ds])
      report['nq'][t][split]=out
     report['shape'][t]={}
     for group,runs in [('row0',ROW0),('row1',ROW1)]:
      ds=[all_data[(r,'distribution_validation')][0][t]['all'] for r in runs]
      out={k:ms([d[k]['mean'] for d in ds]) for k in ['top1_span_probability','top1_group_probability','span_entropy','group_entropy','effective_groups','groups_for_0.5_mass','groups_for_0.9_mass','groups_for_0.99_mass','top_group_representative_share','modes_ge_0.05','second_disjoint_group_mass','mass_overlapping_gold']}
      for k in ['below_0.5','at_least_0.9']:out['top_span_'+k]=ms([d['top1_probability_shares'][k] for d in ds])
      shares=[];multi=[]
      for r in runs:
       vals=list(all_data[(r,'distribution_validation')][1][t].values())
       shares.append(np.mean([v['top_group_representative_share']<.5 for v in vals]))
       multi.append(np.mean([v['modes_ge_0.05']>=2 for v in vals]))
      out['representative_below_half']=ms(shares);out['two_or_more_modes']=ms(multi)
      report['shape'][t][group]=out
    for run in ROW1+ROW0:
     c=all_data[(run,'distribution_validation')][0]['metadata']['map_validation_check']
     report['validation_checks'][run]={'measured':c['measured'],'saved':c['expected'],'delta':{k:c['measured'][k]-c['expected'][k] for k in ['exact','f1']}}
    report['outcome_shape']={}
    for group,runs in [('row0',ROW0),('row1',ROW1)]:
     report['outcome_shape'][group]={}
     for outcome in ['correct','partial','disjoint']:
      ds=[all_data[(r,'distribution_validation')][0]['calibrated']['by_map_outcome'][outcome] for r in runs]
      report['outcome_shape'][group][outcome]={'questions':ms([d['questions'] for d in ds]), **{k:ms([d[k]['mean'] for d in ds]) for k in ['group_entropy','top1_group_probability','groups_for_0.99_mass','second_disjoint_group_mass']}}
    Path('reports/answer_distribution_2026-10-31.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for split in ['all','heldout']:
     print('NQ',split,{k:round(v['mean'],4) for k,v in report['nq']['calibrated'][split].items() if isinstance(v,dict)})
    for group in ['row0','row1']:
     print('SHAPE',group,{k:round(v['mean'],4) for k,v in report['shape']['calibrated'][group].items()})


if __name__ == "__main__":
    main()
