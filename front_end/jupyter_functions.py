import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell

def create_ipynb_from_script(script_path, notebook_path):
    # Read the Python script
    with open(script_path, 'r') as file:
        script_content = file.read()

    # Create a new notebook
    notebook = new_notebook()

    # Create a code cell with the script content
    code_cell = new_code_cell(script_content)

    # Add the code cell to the notebook
    notebook.cells.append(code_cell)

    # Save the notebook to a file
    with open(notebook_path, 'w') as file:
        nbf.write(notebook, file)

script_path = __file__
notebook_path = 'converted_script.ipynb'

create_ipynb_from_script(script_path, notebook_path)

import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell

import voila
import asyncio

async def run_voila(notebook_path):
    await voila.app.launch_new_instance(['--no-browser', notebook_path])

notebook_path = 'path_to_your_notebook.ipynb'

loop = asyncio.get_event_loop()
loop.run_until_complete(run_voila(notebook_path))



