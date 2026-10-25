import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from analyze_all_repeat_trajectories import controls, phase, compare


def test_strict_controls_do_not_relax_block_duration_or_content():
    target=dict(id='t',session='d',block=2,text='repeat',bins=100,index=0)
    rows=[target]+[dict(id=str(i),session=d,block=b,text=t,bins=n,index=i) for i,d,b,t,n in
        [(1,'d',2,'other',95),(2,'d',2,'other2',110),(3,'d',2,'long',111),
         (4,'d',3,'wrongblock',100),(5,'e',2,'wrongday',100),(6,'d',2,'repeat',100)]]
    assert controls(dict(right='t'),{'t':target},rows,{'repeat'})==['1','2']


def test_phase_sensitivity_and_identical_metric():
    a=np.stack([np.linspace(0,1,20),np.linspace(1,2,20)],axis=1)
    b=np.stack([np.linspace(0,1,40),np.linspace(1,2,40)],axis=1)
    np.testing.assert_allclose(phase(a),phase(b),atol=1e-15)
    assert compare(a,a)['RMS']==0
    assert compare(a,b)['coverage']==[1.,.5]
    assert np.isclose(compare(phase(a),phase(b))['shape_cosine'],1)
