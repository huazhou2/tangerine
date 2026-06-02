"""
Prepare dataset for 6-year survival prediction.
Filters time_at_event < 0, computes y_seq / y_mask, creates chronological splits.

Splits are based on CT scan date (ct_date) with fixed date cutoffs:
  ct_date < 2021-01-01          → train
  2021-01-01 <= ct_date < 2022-01-01  → val
  ct_date >= 2022-01-01         → test

Usage:
    python prepare_survival_dataset.py \
        --metadata_csv  /path/to/lungct_with_mrn_anonacc.csv \
        --images_dir    /path/to/images_3d_swine \
        --output_dir    ./dataset_survival_splits \
        --max_followup  6
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

MAX_FOLLOWUP = 6


def make_survival_labels(cancer: int, time_years: float, max_followup: int):
    """Return y_seq [T] and y_mask [T] for one patient."""
    y_seq  = np.zeros(max_followup, dtype=np.float32)
    y_mask = np.zeros(max_followup, dtype=np.float32)
    time_at_event = int(min(time_years, max_followup - 1))  # floor to year index

    if cancer == 1:
        y_seq[time_at_event:] = 1.0          # 1 from diagnosis year onward
    # mask: supervise up to time_at_event + 1 years
    y_mask[:time_at_event + 1] = 1.0
    return y_seq, y_mask, time_at_event


def prepare(args):
    print(f"\n{'='*65}")
    print(f"SURVIVAL DATASET PREPARATION")
    print(f"{'='*65}\n")

    df = pd.read_csv(args.metadata_csv)
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(columns=[df.columns[0]])

    print(f"Loaded {len(df)} records from metadata CSV")

    # ── Identify ID column ─────────────────────────────────────────────
    for col in ['ct_id', 'AnonAcc', 'MRN']:
        if col in df.columns:
            id_col = col
            break
    else:
        raise ValueError("No ID column found (ct_id / AnonAcc / MRN)")
    print(f"ID column: {id_col}")

    # ── Filter time_at_event < 0 ───────────────────────────────────────
    before = len(df)
    df = df[df['time_ct_to_last_event_or_followup'] >= 0].copy()
    print(f"After removing time<0: {len(df)} records (removed {before - len(df)})")
    print(f"  Cancer positive: {int(df['cancer'].sum())}")
    print(f"  Cancer negative: {int((df['cancer']==0).sum())}")

    # ── Match with available images ────────────────────────────────────
    images_path = Path(args.images_dir)
    available   = {f.stem.replace('.nii', '')
                   for f in images_path.glob('*.nii.gz')}
    df['image_filename'] = df[id_col].astype(str) + '.nii.gz'
    df = df[df[id_col].astype(str).isin(available)].copy()
    print(f"After image matching:  {len(df)} records")

    # ── Compute survival labels ────────────────────────────────────────
    y_seqs, y_masks, tae_list = [], [], []
    for _, row in df.iterrows():
        ys, ym, tae = make_survival_labels(
            int(row['cancer']),
            float(row['time_ct_to_last_event_or_followup']),
            args.max_followup
        )
        y_seqs.append(ys)
        y_masks.append(ym)
        tae_list.append(tae)

    for t in range(args.max_followup):
        df[f'y_seq_{t}']  = [s[t] for s in y_seqs]
        df[f'y_mask_{t}'] = [m[t] for m in y_masks]
    df['time_at_event'] = tae_list

    # ── Evaluable counts per year ──────────────────────────────────────
    print(f"\nEvaluable patients per year cutoff (after filtering):")
    for yr in range(1, args.max_followup + 1):
        vp = int(((df['cancer']==1) & (df['time_ct_to_last_event_or_followup'] <= yr)).sum())
        vn = int(((df['cancer']==0) & (df['time_ct_to_last_event_or_followup'] >= yr)).sum())
        print(f"  Year {yr}: pos={vp:4d}  neg={vn:5d}  total={vp+vn:5d}")

    # ── Chronological split by ct_date (fixed date cutoffs) ──────────
    if 'ct_date' not in df.columns:
        raise ValueError("ct_date column not found — required for chronological split")
    df['ct_date'] = pd.to_datetime(df['ct_date'], errors='coerce')
    n_invalid = df['ct_date'].isna().sum()
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} records have unparseable ct_date — dropping")
        df = df[df['ct_date'].notna()].copy()

    cut_val  = pd.Timestamp('2021-01-01')
    cut_test = pd.Timestamp('2022-01-01')

    train_df = df[df['ct_date'] <  cut_val].copy()
    val_df   = df[(df['ct_date'] >= cut_val) & (df['ct_date'] < cut_test)].copy()
    test_df  = df[df['ct_date'] >= cut_test].copy()

    print(f"\nChronological splits (fixed date cutoffs):")
    print(f"  train: ct_date < {cut_val.date()}")
    print(f"  val:   {cut_val.date()} <= ct_date < {cut_test.date()}")
    print(f"  test:  ct_date >= {cut_test.date()}")
    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        date_min = d['ct_date'].min().strftime('%Y-%m-%d')
        date_max = d['ct_date'].max().strftime('%Y-%m-%d')
        print(f"  {name}: {len(d):5d}  (cancer={int(d['cancer'].sum())}"
              f"  dates: {date_min} → {date_max})")

    # ── Save ───────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / 'train.csv', index=False)
    val_df.to_csv(out  / 'val.csv',   index=False)
    test_df.to_csv(out / 'test.csv',  index=False)

    config = {
        'split_method': 'chronological_fixed_dates',
        'train_cutoff': str(cut_val.date()),
        'test_cutoff':  str(cut_test.date()),
        'max_followup': args.max_followup,
        'total': len(df),
        'cancer_pos': int(df['cancer'].sum()),
        'cancer_neg': int((df['cancer']==0).sum()),
        'train': len(train_df), 'val': len(val_df), 'test': len(test_df),
        'train_dates': [train_df['ct_date'].min().strftime('%Y-%m-%d'),
                        train_df['ct_date'].max().strftime('%Y-%m-%d')],
        'val_dates':   [val_df['ct_date'].min().strftime('%Y-%m-%d'),
                        val_df['ct_date'].max().strftime('%Y-%m-%d')],
        'test_dates':  [test_df['ct_date'].min().strftime('%Y-%m-%d'),
                        test_df['ct_date'].max().strftime('%Y-%m-%d')],
    }
    with open(out / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata_csv',  required=True)
    p.add_argument('--images_dir',    required=True)
    p.add_argument('--output_dir',    default='./dataset_survival_splits')
    p.add_argument('--max_followup',  type=int,   default=6)
    prepare(p.parse_args())
