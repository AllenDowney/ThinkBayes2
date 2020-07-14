#!/usr/bin/env python
"""
simple example script for scrubping solution code cells from IPython notebooks

Usage: `scrub_code.py foo.ipynb [bar.ipynb [...]]`

Marked code cells are scrubbed from the notebook
"""

import io
import os
import sys

import nbformat

def scrub_code_cells(nb):
    scrubbed = 0
    cells = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        cells += 1
        # scrub cells marked with initial '# Solution' comment
        # any other marker will do, or it could be unconditional
        if cell.source.startswith("# Solution"):
            cell.source = u'# Solution goes here'
            scrubbed += 1

        # clear all outputs
        cell.outputs = []

    print("scrubbed %i/%i code cells" % (scrubbed, cells))

if __name__ == '__main__':
    for filename in sys.argv[1:]:
        print("reading %s" % filename)
        # read
        with io.open(filename, encoding='utf8') as f:
            nb = nbformat.read(f, nbformat.NO_CONVERT)

        # scrub
        scrub_code_cells(nb)

        # new name
        base, ext = os.path.splitext(filename)
        
        if base.endswith("soln"):
            base = base.replace("soln", "")
            base = base.rstrip("_")
            base = '../' + base
        else:
            base = base + "_scrubbed"

        new_filename = "%s%s" % (base, ext)
        
        # write
        print("writing %s" % new_filename)
        with io.open(new_filename, 'w', encoding='utf8') as f:
            nbformat.write(nb, f)
