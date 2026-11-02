"""Exact small-space checks for full-distribution diagnostics (#95/#96)."""
import json

import numpy as np
import pytest

from squadqa.answer_distribution import (aggregate, divergences, grouped, multi_answer,
                                        nq_questions, regions, shape, summary)
from squadqa.decode import SpanScores


def spans(starts, ends, probabilities, null=None):
    return SpanScores(np.array(starts), np.array(ends), np.log(probabilities),
                      None if null is None else float(np.log(null)))


def test_pool_all_boundary_variants_before_entropy_and_concentration():
    # "the A . B": two variants of A pool to .6 and outrank B's .4.
    s = spans([0, 1, 3], [1, 1, 3], [.3, .3, .4])
    flags = np.array([False, True, False, True])
    g = grouped(s, flags, 1)
    assert g['mass'] == pytest.approx([.6, .4])
    r = shape(s, flags, (1, 1), 1)
    assert r['span_entropy'] == pytest.approx(-sum(x*np.log(x) for x in [.3,.3,.4]))
    assert r['group_entropy'] == pytest.approx(-sum(x*np.log(x) for x in [.6,.4]))
    assert r['top_group_representative_share'] == pytest.approx(.5)
    assert r['groups_for_0.5_mass'] == 1
    assert r['groups_for_0.9_mass'] == 2
    assert r['mass_overlapping_gold'] == pytest.approx(.6)
    assert r['mass_disjoint_gold'] == pytest.approx(.4)
    assert r['modes_ge_0.05'] == 2


def test_null_is_not_renormalized_out_of_full_distribution():
    r = shape(spans([0], [0], [.2], null=.8), np.ones(1, bool), None, 1)
    assert r['null_mass'] == pytest.approx(.8)
    assert r['groups_for_0.5_mass'] is None
    assert r['mass_overlapping_gold'] is None
    assert r['span_entropy'] == pytest.approx(-.2*np.log(.2)-.8*np.log(.8))


def test_modes_exclude_overlap_and_noncontent_groups():
    s = spans([0, 0, 2, 3], [1, 0, 2, 3], [.4, .3, .2, .1])
    r = shape(s, np.array([True,True,True,False]), (0,0), 1)
    assert r['modes_ge_0.05'] == 2
    assert r['second_disjoint_group_mass'] == pytest.approx(.2)


def test_region_merge_overlap_not_adjacency():
    rs, q = regions([(0,2),(1,3),(4,4),(9,9)])
    assert rs == [(0,3),(4,4),(9,9)]
    assert q == pytest.approx([.5,.25,.25])


def test_region_mass_excludes_cross_span_and_has_rest_bucket():
    s = spans([0, 3, 0, 6], [0, 3, 3, 6], [.5,.2,.2,.1])
    r = multi_answer(s, np.ones(7,bool), [(0,0),(3,3)], 1)
    assert r['p_regions'] == pytest.approx([.5,.2])
    assert r['outside_mass'] == pytest.approx(.3)
    assert r['collapse'] == pytest.approx(5/7)
    assert r['second_to_first_mass'] == pytest.approx(.4)
    assert r['kl_q_p'] == pytest.approx(.5*np.log(.5/.2))
    # Stable ties: group [0,3] precedes [3,3], so the latter is third, not second.
    assert r['second_region_best_group_rank'] == 3
    assert r['top_region_is_first_annotated']


def test_temperature_uses_same_scores_but_changes_mass():
    s = spans([0,2],[0,2],[.9,.1])
    assert grouped(s,np.ones(3,bool),2)['mass'] == pytest.approx([.75,.25])


def test_divergences_exact_and_zero_model_probability_safe():
    assert divergences(np.array([.5,.5,0]),np.array([.5,.5,0])) == pytest.approx((0,0),abs=1e-7)
    kl, js = divergences(np.array([.5,.5,0]),np.array([1.,0,0]))
    assert np.isfinite(kl) and np.isfinite(js)
    assert 0 <= js <= np.sqrt(np.log(2))


def test_summary_uses_two_region_subset_and_separates_heldout():
    a = multi_answer(spans([0,2],[0,2],[.9,.1]),np.ones(3,bool),[(0,0),(2,2)],1)
    b = multi_answer(spans([0],[0],[1.]),np.ones(1,bool),[(0,0)],1)
    a['split'] = 'heldout'; b['split'] = 'train'
    out = summary({'a':a,'b':b}, 'multi-answer')
    assert out['all']['total'] == 2
    assert out['all']['eligible'] == 1
    assert out['heldout']['eligible'] == 1
    assert out['train']['eligible'] == 0
    assert aggregate([]) == {'questions':0}


def test_nq_rebuild_uses_annotated_offsets_and_collapses_answer_copies(tmp_path):
    qs = {'q_a_inside': {'split':'train','answers':[[0,'A',.5],[4,'A',.5]]},
          'held': {'split':'heldout','answers':[[0,'B',1.]]}}
    (tmp_path/'q.json').write_text(json.dumps(qs))
    for split, data in [('train',[{'id':'q_a_inside_a0','question':'Which?','answers':[]},
                                 {'id':'q_a_inside_a1','question':'Which?','answers':[]}]),
                        ('heldout',[{'id':'held','question':'Which?','answers':[]}])]:
        context = 'A X A' if split == 'train' else 'B'
        (tmp_path/f'{split}.json').write_text(json.dumps({'data':[{'title':split,'paragraphs':[{'context':context,'qas':data}]}]}))
    out, _ = nq_questions(tmp_path)
    assert len(out) == 2
    assert out[0].id == 'q_a_inside'
    assert out[0].answers == (('A',0),('A',4))
