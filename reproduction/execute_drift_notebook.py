"""Execute the plain-Python audit companion without installing Jupyter.

No magics or IPython expressions are supported. Captured stdout and successful
assertions are saved in a separate executed notebook; original is preserved.
"""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); notebook=json.loads(args.input.read_text())
    assert notebook['nbformat']==4 and notebook['nbformat_minor']==5
    assert len({c['id'] for c in notebook['cells']})==len(notebook['cells'])
    namespace={'__name__':'__notebook__'};count=0
    for cell in notebook['cells']:
        assert cell['cell_type'] in ('markdown','code') and isinstance(cell['source'],list)
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);ast.parse(source);count+=1
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(compile(source,str(args.input),'exec'),namespace)
        cell['execution_count']=count
        cell['outputs']=[dict(output_type='stream',name='stdout',text=capture.getvalue().splitlines(True))]
    notebook['metadata']['execution_note']='Plain-Python cells executed top-to-bottom; no Jupyter kernel or magics'
    target=args.input.with_name(args.input.stem+'.executed.ipynb')
    assert not target.exists(), 'Preserve existing executed notebook'
    target.write_text(json.dumps(notebook,indent=1)+'\n')
    print('Notebook executed successfully:',count,'code cells;',target)


if __name__=='__main__':main()
