from types import SimpleNamespace
from research.reporting import resources


def test_resource_sampler_excludes_reused_pids_and_reports_units(monkeypatch):
    output = '42 12 50.0 python -m research.run --stage initial-search\n43 99 0.0 unrelated-process'
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=output,returncode=0))
    rows = resources.native_processes([42,43])
    assert len(rows)==1 and rows[0]['rss_bytes']==12*1024
    assert rows[0]['cpu_percent']==50.


def test_decoder_memory_scope_and_unavailable_observation(monkeypatch):
    output = '100\n200\nanon 60\nfile 30\nkernel 10\n'
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=output,returncode=0))
    result = resources.decoder_memory('fixture')
    assert result['current_bytes']==100 and result['lifetime_peak_bytes']==200
    assert result['anon_bytes']==60 and result['file_cache_bytes']==30
    monkeypatch.setattr(resources.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout='',returncode=1))
    assert not resources.decoder_memory('fixture')['available']
