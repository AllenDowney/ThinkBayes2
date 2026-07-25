import nbformat as nbf
from glob import glob


def process_cell(cell):
    cell_tags = cell.get('metadata', {}).get('tags', [])

    # add hide-cell tag to solutions
    if cell['cell_type'] == 'code':
        source = cell['source']
        if source.startswith('# Solution'):
            tag = 'hide-cell'
            if tag not in cell_tags:
                cell_tags.append('hide-cell')

    if len(cell_tags) > 0:
        cell['metadata']['tags'] = cell_tags


def process_notebook(path):
    ntbk = nbf.read(path, nbf.NO_CONVERT)

    for cell in ntbk.cells:
        process_cell(cell)

    nbf.write(ntbk, path)


# Collect a list of the notebooks in the content folder
paths = glob("chap*.ipynb")

for path in sorted(paths):
    print('prepping', path)
    process_notebook(path)
