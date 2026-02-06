# python
"""
Runner to generate PNG plots for a temperature folder.
Usage: run from project or in PyCharm by running this script.
"""
import os
import sys
from pathlib import Path

# Ensure the data_process package path is importable
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from plot import read_and_plot_processed_plots, aggregate_and_plot_grouped

import argparse


def find_best_source(temp_folder_name: str):
    # Prefer processed CSVs under dataset/LG_HG2_processed/<temp>
    processed = Path(_THIS_DIR) / Path('..') / 'dataset' / 'LG_HG2_processed' / temp_folder_name
    processed = processed.resolve()
    processed_plots = Path(_THIS_DIR) / Path('..') / 'dataset' / 'LG_HG2_processed_plots' / temp_folder_name
    processed_plots = processed_plots.resolve()

    if processed.is_dir() and any(p.suffix == '.csv' for p in processed.iterdir()):
        return str(processed)
    if processed_plots.is_dir() and any(p.suffix == '.csv' for p in processed_plots.iterdir()):
        # Some projects store CSVs in a folder named *_plots; accept it
        return str(processed_plots)
    # fallback: return processed if exists, else processed_plots
    if processed.is_dir():
        return str(processed)
    if processed_plots.is_dir():
        return str(processed_plots)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot processed battery data or aggregate by test type.')
    parser.add_argument('temp', nargs='?', default='25degC', help='Temperature folder name under dataset (e.g., 25degC) or absolute path')
    parser.add_argument('-o', '--out', dest='out_dir', help='Output directory for generated plots')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate same-type CSVs and create combined overlay plots')
    args = parser.parse_args()

    temp = args.temp
    src = find_best_source(temp)
    if src is None:
        print('Could not find processed data for', temp)
        print('Tried these locations:')
        print(' -', (Path(_THIS_DIR) / Path('..') / 'dataset' / 'LG_HG2_processed' / temp).resolve())
        print(' -', (Path(_THIS_DIR) / Path('..') / 'dataset' / 'LG_HG2_processed_plots' / temp).resolve())
        sys.exit(1)

    out_dir = args.out_dir if args.out_dir else str((Path(_THIS_DIR) / Path('..') / 'dataset' / 'plot').resolve())

    if args.aggregate:
        print('Aggregate mode: grouping same-type CSVs and plotting overlays')
        groups, saved = aggregate_and_plot_grouped(src, out_dir=out_dir)
        print(f'Combined groups: {groups}, saved {len(saved)} files under {out_dir}')
        for s in saved:
            print('  -', s)
    else:
        print('Per-file plotting mode')
        total, success = read_and_plot_processed_plots(src, out_dir=out_dir, show=False)
        print(f'Done. total={total}, success_count={len(success)}')
        for p in success:
            print('  -', p)
