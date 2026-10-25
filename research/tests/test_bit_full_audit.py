"""Independent standard CTC accounting for E016's later final audit."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'reproduction'))
from verify_bit_full import distance,decode,decode_source_order


def test_independent_distances():
    assert distance([],[])==0
    assert distance([1,2],[1,3,4])==2
    assert distance([1,1],[1])==1


def test_native_blank_separates_repeat():
    values=np.full((5,41),-20.,np.float32)
    values[np.arange(5),[1,1,40,1,39]]=0
    assert decode(values)==[1,1,39]
    assert decode_source_order(values)==[1,1,39]


def test_export_permutation_preserves_source_blank_tie_priority():
    values=np.full((3,41),-20.,np.float32)
    values[np.arange(3),[2,40,2]]=0
    values[1,2]=0  # Blank and phone tie: author blank=0 wins.
    assert decode(values)==[2]
    assert decode_source_order(values)==[2,2]


def test_source_phone_tie_priority_unchanged():
    values=np.full((1,41),-20.,np.float32)
    values[0,[3,5]]=0
    assert decode_source_order(values)==decode(values)==[3]
