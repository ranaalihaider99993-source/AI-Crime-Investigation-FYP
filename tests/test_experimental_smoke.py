import pytest
from scripts.experimental_smoke import eligible_rows, run_experimental_smoke


def test_filter_preserves_split_and_excludes_unverified():
    rows=[dict(source=str(i),split='val',label='Vandalism') for i in range(5)]
    answers={str(i):dict(label_visible=status,videos_watched='Confirmed',incident_independence='Pending')
             for i,status in enumerate(['Confirmed','Uncertain','Rejected','Pending','Confirmed'])}
    answers['4']['videos_watched']='Pending'
    included,excluded=eligible_rows(rows,answers)
    assert included==rows[:1] and len(excluded)==4
    assert answers['0']['incident_independence']=='Pending'


def test_experiment_requires_separate_authorization():
    with pytest.raises(ValueError,match='authorization'):
        run_experimental_smoke(None,None,None)


def test_uncertain_opt_in_still_excludes_rejected_pending_unwatched():
    rows=[dict(source=str(i),split='test') for i in range(5)]
    answers={str(i):dict(label_visible=status,videos_watched='Confirmed')
             for i,status in enumerate(['Confirmed','Uncertain','Rejected','Pending','Uncertain'])}
    answers['4']['videos_watched']='Pending'
    included,excluded=eligible_rows(rows,answers,allow_uncertain_labels=True)
    assert included==rows[:2] and len(excluded)==3
