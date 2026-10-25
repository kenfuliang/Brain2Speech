from reproduction.summarize_causal_recipe_evaluation import paired,target_budget,aggregate


def rows():
    return [dict(id=str(i),session='d'+str(i//2),**{f:dict(errors=i%2,reference_units=4,substitutions=i%2,deletions=0,insertions=0) for f in ('wer','per','cer')}) for i in range(4)]


def test_identical_paired_interval_is_zero():
    r=rows();result=paired(r,r,repetitions=50)
    assert result['wer']==dict(anscombe_minus_raw=0.,ci95=[0.,0.])


def test_count_aggregation_not_trial_rate_average():
    r=rows();r[0]['wer']['reference_units']=100
    assert aggregate(r)['wer']['corpus']==2/112


def test_target_is_first_measured_crossing_or_not_reached():
    assert target_budget([dict(labels=0,wer=.7),dict(labels=5,wer=.24)],.25)==5
    assert target_budget([dict(labels=400,wer=.26)],.25)=='not reached'
