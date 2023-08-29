from json import dump
from pathlib import Path
from typing import Dict, Any


def save_results(results: Dict[str, Dict[str, Any]], model_name: str):
    results_dir = Path.cwd() / "results"
    results_dir.mkdir(exist_ok=True)

    filepath = results_dir / f"{model_name}.json"

    with filepath.open(mode='w') as fp:
        dump(results, fp, indent=4)
