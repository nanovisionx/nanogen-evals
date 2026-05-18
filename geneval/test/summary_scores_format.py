"""
Example:
python test/summary_scores_format.py assets/geneval-FLUX.1-Krea-dev-cfg3.5-steps30-res1024-seed42_ours.jsonl
"""

import argparse
import pandas as pd

import subprocess
import re
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--outdir", type=str, default="assets")
    args = parser.parse_args()

    assert args.filename.endswith(".jsonl"), "Input file must be a JSONL file"
    try:
        result = subprocess.run(
            ["python", f"test/summary_scores.py", args.filename],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Command completed successfully")
    except subprocess.CalledProcessError as e:
        print("Command failed.")
        print("Error:", e.stderr)
        print("Exit code:", e.returncode)

    task_re = re.compile(
        r'(?P<task>\w+)\s*=\s*'
        r'(?P<percent>\d+\.\d+)%\s*'
    )
    tasks = {}
    for m in task_re.finditer(result.stdout):
        name = m.group('task')
        pct  = float(m.group('percent'))
        tasks[name] = pct / 100
    overall_re = re.search(r'Overall score.*?:\s*(?P<score>[\d.]+)', result.stdout)
    tasks['overall'] = float(overall_re.group('score'))

    tasks = {k: [v] for k, v in tasks.items()}
    tasks = pd.DataFrame(tasks)
    os.makedirs(args.outdir, exist_ok=True)
    tasks.to_csv(os.path.join(args.outdir, os.path.basename(args.filename).replace(".jsonl", ".csv")), index=False)
