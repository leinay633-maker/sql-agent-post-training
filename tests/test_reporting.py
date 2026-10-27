"""Report accounting tests use explicit in-memory test records, not experiments."""
import copy
from pathlib import Path
import pytest
from exec_feedback.core import digest, write_jsonl
from exec_feedback.evaluation import summarize
from exec_feedback.reporting import compare_runs
from exec_feedback.storage import atomic_json


def fixture_run(path,correct):
    rows=[{'task_id':str(i),'db_id':'db'+str(i),'correct':v,'status':'correct' if v else 'incorrect',
           'fixture':False,'steps':[],'final_sql':'SELECT 1','usage':{}} for i,v in enumerate(correct)]
    # Deliberately controlled accounting inputs; no model is called and these
    # files live only in pytest's temporary directory. Never publish their scores.
    identity={'id':'unit-test-records','revision':'unit-test','temperature':0,'seed':42}
    write_jsonl(path/'episodes.jsonl',rows)
    atomic_json(path/'manifest.json',{'task_ids':[r['task_id'] for r in rows],'dataset_id':'test','code':'test',
                'limits':{},'provider':identity,'max_steps':4,'feedback':True})
    atomic_json(path/'summary.json',summarize(rows,len(rows),identity))
    atomic_json(path/'completed.json',{'episodes_sha256':digest(path/'episodes.jsonl'),'summary_sha256':digest(path/'summary.json')})


def test_report_is_complete_and_escapes_untrusted_text(tmp_path):
    fixture_run(tmp_path/'a',[True,False]);fixture_run(tmp_path/'b',[True,True])
    report=compare_runs({'<script>':tmp_path/'a','repair':tmp_path/'b'},tmp_path/'report')
    assert report['tasks']==2
    assert report['paired']['<script> -> repair']['delta_right_minus_left']==.5
    page=(tmp_path/'report/report.html').read_text(encoding="utf-8")
    assert '<script>' not in page and '&lt;script&gt;' in page
    assert all((tmp_path/'report'/name).exists() for name in ['comparison.json','report.md','report.html','per-task.csv'])


def test_report_rejects_modified_episodes(tmp_path):
    fixture_run(tmp_path/'a',[True,False]);fixture_run(tmp_path/'b',[True,True])
    (tmp_path/'b/episodes.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='modified'):compare_runs({'a':tmp_path/'a','b':tmp_path/'b'},tmp_path/'r')


def test_different_budgets_cannot_be_attributed_to_training(tmp_path):
    from exec_feedback.storage import read_json
    fixture_run(tmp_path/'a',[True,False]);fixture_run(tmp_path/'b',[True,True])
    m=read_json(tmp_path/'a/manifest.json');m.update(max_steps=1,feedback=False);atomic_json(tmp_path/'a/manifest.json',m)
    r=compare_runs({'a':tmp_path/'a','b':tmp_path/'b'},tmp_path/'r')
    assert not r['paired']['a -> b']['training_attribution_allowed_by_budget']


def test_model_settings_must_match(tmp_path):
    from exec_feedback.storage import read_json
    fixture_run(tmp_path/'a',[True,False]);fixture_run(tmp_path/'b',[True,True])
    m=read_json(tmp_path/'b/manifest.json');m['provider']['seed']=43;atomic_json(tmp_path/'b/manifest.json',m)
    with pytest.raises(ValueError,match='settings differ'):compare_runs({'a':tmp_path/'a','b':tmp_path/'b'},tmp_path/'r')
