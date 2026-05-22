"""Scrape per-task success from each WGSP eval row's results.txt and emit a
single CSV (thesis/wgsp_results.csv) with one row per (ablation, seed)
plus a final block of per-ablation means/stds across seeds.
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path


# Match: "  task_name: 23.4%"
LINE_RE = re.compile(r'^\s+(?P<name>[^\s:][^:]*):\s+(?P<pct>[\d.]+)%\s*$')


def parse_results_txt(path):
    """Parse a results.txt → dict {task_name: success_rate_in_[0,1]}.

    Returns {} if the file is malformed or missing 'overall'."""
    per_task = {}
    if not path.exists():
        return per_task
    in_block = False
    for line in path.read_text().splitlines():
        if line.startswith('===='):
            in_block = True
            per_task = {}
            continue
        if not in_block:
            continue
        m = LINE_RE.match(line)
        if m:
            per_task[m.group('name').strip()] = float(m.group('pct')) / 100.0
    return per_task


def aggregate(results_root, output, expected_rows=None):
    rows = []
    by_tag = defaultdict(list)

    for results_dir in sorted(Path(results_root).iterdir()):
        if not results_dir.is_dir():
            continue
        m = re.match(r'row(?P<row>\d+)_(?P<tag>[^_]+(?:_[^_]+)*?)_s(?P<seed>\d+)$',
                     results_dir.name)
        if not m:
            continue
        row_id = int(m.group('row'))
        tag    = m.group('tag')
        seed   = int(m.group('seed'))

        # eval_ogbench.py writes results.txt inside the dir.
        rt = results_dir / 'results.txt'
        per_task = parse_results_txt(rt)
        if not per_task or 'overall' not in per_task:
            print(f"  skipping {results_dir.name}: no parsed overall")
            continue

        rows.append({
            'row': row_id, 'tag': tag, 'seed': seed,
            **per_task,
        })
        by_tag[(row_id, tag)].append(per_task['overall'])

    if not rows:
        print(f"No results parsed under {results_root}.")
        return

    # Stable column order
    cols = ['row', 'tag', 'seed'] + sorted({c for r in rows for c in r if c not in
                                            ('row', 'tag', 'seed')})

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x['row'], x['seed'])):
            w.writerow({c: r.get(c, '') for c in cols})

        # Aggregated rows: mean / std of overall per (row, tag).
        f.write('\n# aggregated\n')
        agg = csv.writer(f)
        agg.writerow(['row', 'tag', 'n_seeds', 'overall_mean', 'overall_std'])
        for (row_id, tag), vals in sorted(by_tag.items()):
            n = len(vals)
            mean = sum(vals) / n
            std  = (sum((v - mean) ** 2 for v in vals) / max(n - 1, 1)) ** 0.5
            agg.writerow([row_id, tag, n, f'{mean:.4f}', f'{std:.4f}'])

    print(f"Wrote {output} ({len(rows)} per-seed rows, "
          f"{len(by_tag)} aggregated configurations).")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', default='./eval_wgsp')
    parser.add_argument('--output',       default='thesis/wgsp_results.csv')
    args = parser.parse_args()
    aggregate(args.results_root, args.output)
