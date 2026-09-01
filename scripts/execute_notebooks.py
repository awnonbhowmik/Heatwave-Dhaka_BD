#!/usr/bin/env python3
"""Execute every thin notebook from a restarted kernel."""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
for path in sorted((ROOT/"notebooks").glob("*.ipynb")):
    nb=nbformat.read(path,as_version=4)
    NotebookClient(nb,timeout=120,kernel_name="python3",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,path)
    print(f"executed {path.name}")
