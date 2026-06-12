"""
Prepare dataset for 6-year survival prediction.
Filters time_at_event < 0, computes y_seq / y_mask, creates stratified splits.

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
from sklearn.model_selection import train_test_split

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

    # ── Stratified split at PATIENT level (prevent data leakage) ─────────
    print(f"\n{'-'*65}")
    print("PATIENT-LEVEL SPLIT (preventing same patient in multiple splits)")
    print(f"{'-'*65}")

    # Get unique patients with their cancer status (use first occurrence)
    # CRITICAL: Must group by PatientID, not ct_id (ct_id is scan ID, PatientID is patient ID)
    patient_df = df.groupby('PatientID').agg({
        'cancer': 'first',  # Assume cancer status is same for all CTs of a patient
        'ct_id': 'count'    # Count number of CTs per patient
    }).rename(columns={'ct_id': 'num_cts'}).reset_index()

    print(f"Unique patients: {len(patient_df)}")
    print(f"  Cancer positive: {int(patient_df['cancer'].sum())}")
    print(f"  Cancer negative: {int((patient_df['cancer']==0).sum())}")

    # Stratified split at patient level
    train_pats, temp_pats = train_test_split(
        patient_df, test_size=(args.val_ratio + args.test_ratio),
        stratify=patient_df['cancer'], random_state=args.seed
    )
    val_ratio_adj = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_pats, test_pats = train_test_split(
        temp_pats, test_size=(1 - val_ratio_adj),
        stratify=temp_pats['cancer'], random_state=args.seed
    )

    print(f"\nPatient splits:")
    for name, pats in [('Train', train_pats), ('Val', val_pats), ('Test', test_pats)]:
        n_pos = int(pats['cancer'].sum())
        n_neg = len(pats) - n_pos
        print(f"  {name}: {len(pats):4d} patients  (cancer+={n_pos:3d}  cancer-={n_neg:4d})")

    # Map CTs back to splits based on patient assignment
    # Use PatientID (the patient identifier), not ct_id (scan ID)
    train_pats_ids = set(train_pats['PatientID'])
    val_pats_ids = set(val_pats['PatientID'])
    test_pats_ids = set(test_pats['PatientID'])

    train_df = df[df['PatientID'].isin(train_pats_ids)].copy()
    val_df = df[df['PatientID'].isin(val_pats_ids)].copy()
    test_df = df[df['PatientID'].isin(test_pats_ids)].copy()

    print(f"\nCT-level splits (after patient-level assignment):")
    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {name}: {len(d):5d} CTs  (cancer={int(d['cancer'].sum())})")

    # ── Save ───────────────────────────────────────────────────────────
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out / 'train.csv', index=False)
    val_df.to_csv(out  / 'val.csv',   index=False)
    test_df.to_csv(out / 'test.csv',  index=False)

    config = {
        'max_followup': args.max_followup,
        'split_method': 'patient-level',  # ← Indicates data leakage prevention
        'total_patients': len(patient_df),
        'total_cts': len(df),
        'cancer_pos': int(df['cancer'].sum()),
        'cancer_neg': int((df['cancer']==0).sum()),
        'train_patients': len(train_pats), 'train_cts': len(train_df),
        'val_patients': len(val_pats), 'val_cts': len(val_df),
        'test_patients': len(test_pats), 'test_cts': len(test_df),
        'seed': args.seed,
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
    p.add_argument('--train_ratio',   type=float, default=0.70)
    p.add_argument('--val_ratio',     type=float, default=0.15)
    p.add_argument('--test_ratio',    type=float, default=0.15)
    p.add_argument('--seed',          type=int,   default=42)
    prepare(p.parse_args())
