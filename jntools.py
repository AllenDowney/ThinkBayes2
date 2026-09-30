import ast
import sys
import astor
from pathlib import Path
from typing import List
import nbformat as nbf
import typer
import shelve

app = typer.Typer()



@app.command()
def replace_pattern(filenames: List[Path]):
    """ Processes a list of file paths and applies pattern replacement based on the file type.

    For Jupyter Notebook files (.ipynb), it calls the 
    `replace_pattern_in_notebook` function. For Python module files (.py), 
    it calls the `replace_pattern_in_module` function.

    Args:
        filenames (List[Path]): A list of file paths to process. Each file 
        should have a suffix of either '.ipynb' or '.py'.
    """ 
    for filename in filenames:
        if filename.suffix == ".ipynb":
            replace_pattern_in_notebook(filename)
        elif filename.suffix == ".py":
            replace_pattern_in_module(filename)


def replace_pattern_in_notebook(filename):
    """ Processes a Jupyter Notebook file and applies pattern replacement.
    
    Args:
        filename (Path): The path to the Jupyter Notebook file to process.
    """
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            cell.source = replace_pattern_in_source(source)
        elif cell["cell_type"] == "markdown":
            source = cell["source"]
            cell.source = replace_pattern_in_markdown(source)

    nbf.write(notebook, filename)


def replace_pattern_in_module(filename):
    """ Processes a Python module file and applies pattern replacement.

    Args:
        filename (Path): The path to the Python module file to process.
    """
    source = filename.read_text()
    source = replace_pattern_in_source(source)
    filename.write_text(source)


def replace_pattern_in_source(source):
    return source


def replace_pattern_in_markdown(source):
    """ Processes a markdown cell in a Jupyter Notebook and applies pattern replacement.
    Args:
        source (str): The source code of the markdown cell.
    Returns:
        str: The modified source code with patterns replaced.
    """
    pattern = r"(?<!`)(Series|DataFrame|Index)(?!`)"
    replacement = r"`\1`"
    source = re.sub(pattern, replacement, source)

    pattern = r"(?<!`)(Hist|Pmf|Cdf|Surv|Hazard)(?!`)"
    replacement = r"`\1`"
    source = re.sub(pattern, replacement, source)

    pattern = r"(?<!`)(Normal|NormalPdf|EstimatedPdf|HypothesisTest)(?!`)"
    replacement = r"`\1`"
    source = re.sub(pattern, replacement, source)

    pattern = r"pandas"
    replacement = r"Pandas"
    source = re.sub(pattern, replacement, source)

    return source


def put_each_sentence_on_a_new_line(source):
    """ Processes a string and places each sentence on a new line.
    Args:
        source (str): The source string to process.
    Returns:
        str: The modified string with each sentence on a new line.
    """
    # replace a single newline with a space
    pattern = r"(?<!\n)\n(?!\n)"
    replacement = r" "
    source = re.sub(pattern, replacement, source)

    # replace four spaces with one
    pattern = r"     "
    replacement = r" "
    source = re.sub(pattern, replacement, source)

    # split between sentences
    pattern = r"(?<=[^\s.]{3})\.[ ]+"
    replacement = r".\n"
    source = re.sub(pattern, replacement, source)

    return source


@app.command()
def format_sentences(filenames: List[Path]):
    for filename in filenames:
        format_sentences_in_notebook(filename)


def format_sentences_in_notebook(filename):
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell["cell_type"] == "markdown":
            cell.source = put_each_sentence_on_a_new_line(cell.source)
    nbf.write(notebook, filename)


@app.command()
def replace_functions(filenames: List[Path]):
    with shelve.open("function_names") as db:
        function_names = db["function_names"]

    for filename in filenames:
        if filename.suffix == ".ipynb":
            replace_functions_in_notebook(filename, function_names)
        elif filename.suffix == ".py":
            replace_functions_in_module(filename, function_names)


def replace_functions_in_notebook(filename, function_names):
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            modified_source = replace_functions_in_source(source, function_names)
            cell.source = modified_source.rstrip()
    nbf.write(notebook, filename)


def replace_functions_in_module(filename, function_names):
    source = filename.read_text()
    source = replace_functions_in_source(source, function_names)
    filename.write_text(source)


def replace_functions_in_source(source, function_names):
    class_names = [
        "Cdf",
        "Pmf",
        "Hist",
        "Pdf",
        "HazardFunction",
        "SurvivalFunction",
        "HypothesisTest",
    ]

    class FunctionNameReplacer(ast.NodeTransformer):

        def visit_FunctionDef(self, node):
            if node.name in function_names:
                node.name = camel_to_snake(node.name)
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                module_name = node.func.value.id
                function_name = node.func.attr
                if module_name == "thinkstats2" and function_name in class_names:
                    return node
                if function_name in function_names:
                    node.func.attr = camel_to_snake(function_name)
            elif isinstance(node.func, ast.Name):
                if node.func.id in function_names:
                    node.func.id = camel_to_snake(node.func.id)
            self.generic_visit(node)
            return node

    try:
        tree = ast.parse(source)
        replacer = FunctionNameReplacer()
        tree = replacer.visit(tree)
        return astor.to_source(tree)
    except SyntaxError as e:
        print(f"Syntax error in code:\n{source}\nError: {e}")
        return source


@app.command()
def get_functions(filenames: List[Path]):
    function_names = set()
    for filename in filenames:
        if filename.suffix == ".ipynb":
            print(f"Getting functions from {filename}")
            notebook = nbf.read(filename, nbf.NO_CONVERT)
            for cell in notebook.cells:
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"])
                    extract_function_names(source, function_names)
        elif filename.suffix == ".py":
            print(f"Getting functions from {filename}")
            source = filename.read_text()
            extract_function_names(source, function_names)
    with shelve.open("function_names") as db:
        db["function_names"] = function_names
    return function_names


def to_camelcase(s):
    parts = s.split("_")
    return parts[0] + "".join((word.capitalize() for word in parts[1:]))


def is_camelcase(s):
    return s[0].isupper() and s == to_camelcase(s)


import re


def camel_to_snake(s):
    return re.sub("(?<!^)(?=[A-Z])", "_", s).lower()


def extract_function_names(source, function_names):

    class FunctionNameExtractor(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            if is_camelcase(node.name):
                function_names.add(node.name)
            self.generic_visit(node)

    try:
        tree = ast.parse(source)
        extractor = FunctionNameExtractor()
        extractor.visit(tree)
    except SyntaxError as e:
        print(f"Syntax error in code:\n{source}\nError: {e}")
    for name in list(function_names):
        if is_camelcase(name):
            print(name, camel_to_snake(name))


@app.command()
def add_header(header_filename: Path, filenames: List[Path]):
    """
    Add a header to the beginning of every notebook
    """
    typer.echo(f"Header file: {header_filename}")
    typer.echo(f"Target files: {filenames}")
    header = nbf.read(header_filename, nbf.NO_CONVERT)
    for filename in filenames:
        typer.echo(f"Adding {header_filename} to {filename}")
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = header.cells + notebook.cells
        nbf.write(notebook, filename)


@app.command()
def add_footer(footer_filename: Path, filenames: List[Path]):
    """
    Add a footer to the end of every notebook
    """
    typer.echo(f"Footer file: {footer_filename}")
    typer.echo(f"Target files: {filenames}")
    footer = nbf.read(footer_filename, nbf.NO_CONVERT)
    for filename in filenames:
        typer.echo(f"Adding {footer_filename} to {filename}")
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells + footer.cells
        nbf.write(notebook, filename)


@app.command()
def replace_header(header_filename: Path, filenames: List[Path],
                   marker: str = "You can order print"):
    """Replace an existing header cell, matching on content rather than position.

    Looks for the first cell whose source starts with `marker` and replaces it
    with the cells from `header_filename`. Notebooks without a matching cell are
    left alone and reported, so this is safe to run over a directory that mixes
    notebooks with and without headers.
    """
    header = nbf.read(header_filename, nbf.NO_CONVERT)
    replaced, skipped = [], []

    for filename in filenames:
        notebook = nbf.read(filename, nbf.NO_CONVERT)

        index = None
        for i, cell in enumerate(notebook.cells):
            if cell["cell_type"] == "markdown" and cell["source"].startswith(marker):
                index = i
                break

        if index is None:
            skipped.append(filename)
            continue

        notebook.cells[index:index + 1] = header.cells
        nbf.write(notebook, filename)
        replaced.append(filename)

    typer.echo(f"Replaced the header in {len(replaced)} notebooks")
    if skipped:
        typer.echo(f"No header found in {len(skipped)} notebooks (left alone):")
        for filename in skipped:
            typer.echo(f"    {filename}")


@app.command()
def remove_header(n: int, filenames: List[Path]):
    """Remove n cells from the beginning of every notebook"""
    if n == 0:
        print("Removing 0 cells. Nothing to do.")
        return
    for filename in filenames:
        print("Removing first", n, "cells from", filename)
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells[n:]
        nbf.write(notebook, filename)


@app.command()
def remove_footer(n: int, filenames: List[Path]):
    """Remove n cells from the end of every notebook"""
    if n == 0:
        print("Removing 0 cells. Nothing to do.")
        return
    for filename in filenames:
        print("Removing last", n, "cells from", filename)
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        notebook.cells = notebook.cells[:-n]
        nbf.write(notebook, filename)


@app.command()
def remove_cell(index: int, filenames: List[Path]):
    """Remove a single cell at the specified index from every notebook"""
    for filename in filenames:
        print(f"Removing cell at index {index} from {filename}")
        notebook = nbf.read(filename, nbf.NO_CONVERT)
        
        if index < 0 or index >= len(notebook.cells):
            print(f"Warning: Index {index} is out of range for {filename} (has {len(notebook.cells)} cells)")
            continue
            
        # Remove the cell at the specified index
        notebook.cells.pop(index)
        nbf.write(notebook, filename)


@app.command()
def prepare_latex(filenames: List[Path]):
    for filename in filenames:
        print(f"Preparing {filename} for LaTeX")
        process_notebook(filename, process_cell_latex)


def process_notebook(filename, cell_func):
    notebook = nbf.read(filename, nbf.NO_CONVERT)
    for cell in notebook.cells:
        cell_func(cell)
    nbf.write(notebook, filename)


def process_cell_latex(cell):
    tags = cell["metadata"].get("tags", [])
    if cell["cell_type"] == "code":
        source = cell["source"]
        if source.startswith("# Solution"):
            tag = "hide-cell"
            if tag not in tags:
                tags.append(tag)
        if source.startswith("%%expect"):
            t = source.split("\n")[1:]
            cell["source"] = "\n".join(t)
    for tag in tags:
        if tag.startswith("chapter") or tag.startswith("section"):
            print(tag)
            label = f"({tag})=\n"
            cell["source"] = label + cell["source"]
    if len(tags) > 0:
        cell["metadata"]["tags"] = tags


if __name__ == "__main__":
    app()
