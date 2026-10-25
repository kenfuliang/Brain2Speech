"""Contract checks against the pinned authors' code, without loading TensorFlow."""
import ast
import numpy as np
from research.audit import ROOT
from research.data import PHONES
from research.experiment.evaluator import SCORING


def test_phoneme_inventory_matches_pinned_release():
    path = ROOT/'NeuralDecoder/neuralDecoder/datasets/speechDataset.py'
    tree = ast.parse(path.read_text())
    assignment = next(node for node in tree.body if isinstance(node,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='PHONE_DEF_SIL' for t in node.targets))
    assert ast.literal_eval(assignment.value)==PHONES


def test_fixed_decoder_column_order_matches_authors_rearrangement():
    path = ROOT/'NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                    and node.name=='rearrange_speech_logits')
    # Only the reviewed pure NumPy rearrangement function is executed, not module imports.
    scope = {'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    logits = np.arange(2*3*41).reshape(2,3,41)
    expected = scope['rearrange_speech_logits'](logits,has_sil=True)
    np.testing.assert_array_equal(logits[:,:,SCORING['phone_order']],expected)
