"""TANGERINE Longitudinal Heatmap Generator
Usage:
  python generate_heatmap.py --patient_type cancer_only --year 1
  python generate_heatmap.py --patient_type all_patients --year 1-6
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import argparse


def load_data(parent_dir):
    """Load test set, predictions, and LRADS data."""
    test_df = pd.read_csv(os.path.join(parent_dir, 'dataset_splits/test.csv'))
    pred_df = pd.read_csv(os.path.join(parent_dir, 'results_20260605/test_predictions.csv'))
    scan_master = pd.read_csv(os.path.join(parent_dir, 'scan_master_with_lrads_value_v3_with_base.csv'))

    pred_df_renamed = pred_df.rename(columns={'patient_id': 'ct_id'})
    data = test_df.merge(pred_df_renamed, on='ct_id', how='inner', suffixes=('_test', '_pred'))
    data['ct_date'] = pd.to_datetime(data['ct_date'])
    data['first_lung_ca_date'] = pd.to_datetime(data['first_lung_ca_date'])

    lrads_col = 'lrads_category_base' if 'lrads_category_base' in scan_master.columns else 'lrads_category'
    lrads_data = scan_master[['ct_id', lrads_col]].drop_duplicates()
    lrads_data = lrads_data.rename(columns={lrads_col: 'lrads'})
    data = data.merge(lrads_data, on='ct_id', how='left')

    patients_multi = data.groupby('pat_id').size()
    patients_multi = patients_multi[patients_multi > 1].index
    data_multi = data[data['pat_id'].isin(patients_multi)].copy()

    return data_multi


def create_heatmap_data(patient_ids, all_months_list, data_source):
    """Create heatmap arrays: predictions, LRADS, diagnosis markers."""
    heatmap_pred = []
    heatmap_lrads = []
    diagnosis_month_marker = []

    for pat_id in patient_ids:
        patient_scans = data_source[data_source['pat_id'] == pat_id].copy()
        patient_scans = patient_scans.sort_values('ct_date')
        diagnosis_date = patient_scans['first_lung_ca_date'].iloc[0]

        pred_dict = {}
        lrads_dict = {}
        for _, row in patient_scans.iterrows():
            month = row['year_month']
            pred_dict[month] = row['pred']
            lrads_dict[month] = row['lrads'] if pd.notna(row['lrads']) else np.nan

        diagnosis_month = diagnosis_date.to_period('M') if pd.notna(diagnosis_date) else None

        pred_row = []
        lrads_row = []
        diag_data = []
        for month in all_months_list:
            pred_row.append(pred_dict.get(month, np.nan))
            lrads_row.append(lrads_dict.get(month, np.nan))
            diag_data.append(month == diagnosis_month)

        heatmap_pred.append(pred_row)
        heatmap_lrads.append(lrads_row)
        diagnosis_month_marker.append(diag_data)

    return np.array(heatmap_pred), np.array(heatmap_lrads), np.array(diagnosis_month_marker, dtype=bool)


def generate_heatmap(patient_type, year, parent_dir):
    """Generate heatmap for given patient type and year."""

    # Load data
    data_multi = load_data(parent_dir)

    # Prepare data
    data_multi['year_month'] = data_multi['ct_date'].dt.to_period('M')
    data_multi['pred'] = data_multi[f'pred_{year}_pred']

    # Get patients
    if patient_type == 'cancer_only':
        cancer_patients_list = data_multi[data_multi['cancer_pred'] == 1]['pat_id'].unique()
        data_plot = data_multi[data_multi['pat_id'].isin(cancer_patients_list)].copy()
        patient_ids = sorted(cancer_patients_list)
        figsize = (60, 12)
        y_colors = ['red'] * len(patient_ids)
        ylabel = 'Patient ID (Cancer Diagnosed)'
        title_prefix = 'Cancer-Only'
    else:  # all_patients
        all_patients_sorted = sorted(data_multi['pat_id'].unique())
        cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 1])
        non_cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 0])
        patient_ids = cancer_patients + non_cancer_patients
        data_plot = data_multi
        figsize = (60, 333)
        y_colors = ['red'] * len(cancer_patients) + ['black'] * len(non_cancer_patients)
        ylabel = 'Patient ID'
        title_prefix = 'All Patients'

    # Create time grid
    min_date = data_plot['ct_date'].min()
    max_date = data_plot['ct_date'].max()
    max_diagnosis = data_plot['first_lung_ca_date'].max()
    if pd.notna(max_diagnosis) and max_diagnosis > max_date:
        max_date = max_diagnosis

    all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')

    print("="*80)
    print(f"TANGERINE {title_prefix.upper()} LONGITUDINAL HEATMAP — YEAR {year}")
    print("="*80)
    if patient_type == 'cancer_only':
        print(f"Cancer patients: {len(patient_ids)}")
    else:
        print(f"Cancer patients: {len(cancer_patients)}")
        print(f"Non-cancer patients: {len(non_cancer_patients)}")
    print(f"Total: {len(patient_ids)}")
    print(f"Time range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

    # Create heatmap data
    month_labels = [str(m) for m in all_months]
    cmap = plt.cm.RdYlGn_r

    pred_data, lrads_data, diagnosis_data = create_heatmap_data(patient_ids, all_months, data_plot)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pred_data, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

    # Overlay LRADS and diagnosis markers
    for i in range(len(patient_ids)):
        for j in range(len(all_months)):
            lrads_val = lrads_data[i, j]
            if not np.isnan(lrads_val):
                ax.text(j, i, f'{int(lrads_val)}', color='black', fontsize=28, ha='center', va='center',
                        fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1),
                        zorder=8)
            if diagnosis_data[i, j]:
                ax.text(j, i, '*', color='purple', fontsize=70, ha='center', va='center',
                        fontweight='bold', zorder=10)

    # Add separator line for all_patients
    if patient_type == 'all_patients' and len(cancer_patients) > 0:
        ax.axhline(len(cancer_patients) - 0.5, color='black', linewidth=2, linestyle='--', zorder=5)

    # Labels and ticks
    ax.set_xticks(np.arange(0, len(all_months), 6))
    ax.set_xticklabels([month_labels[i] for i in np.arange(0, len(all_months), 6)], rotation=45, ha='right', fontsize=16)
    ax.set_yticks(np.arange(len(patient_ids)))
    y_labels = [str(p) for p in patient_ids]
    ax.set_yticklabels(y_labels, fontsize=16)
    for tick, color in zip(ax.get_yticklabels(), y_colors):
        tick.set_color(color)
        tick.set_fontweight('bold')

    ax.set_xlabel('Year-Month', fontsize=18, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

    # Title
    if patient_type == 'cancer_only':
        title = f'TANGERINE Cancer-Only Predictions (Year {year}): n={len(patient_ids)} patients\n'
    else:
        title = f'TANGERINE All Patients Year {year}: n={len(patient_ids)} (Cancer={len(cancer_patients)}, Non-Cancer={len(non_cancer_patients)})\n'
    title += f'Cell color = Year-{year} cancer probability | PURPLE * = Diagnosis month'
    ax.set_title(title, fontsize=32, fontweight='bold', pad=15)

    # Grid and border
    for i in range(len(all_months)+1):
        ax.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.2)
    rect = Rectangle((-0.5, -0.5), len(all_months), len(patient_ids),
                    linewidth=3, edgecolor='black', facecolor='none', zorder=5)
    ax.add_patch(rect)

    cbar = plt.colorbar(im, ax=ax, label=f'Year-{year} Cancer Probability')

    plt.tight_layout()
    heatmap_dir = os.path.join(parent_dir, 'results_20260605/heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)

    filename = f'tangerine_{patient_type}_year{year}.pdf'
    filepath = os.path.join(heatmap_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()
    print(f"✅ Year {year} complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TANGERINE longitudinal heatmaps')
    parser.add_argument('--patient_type', choices=['cancer_only', 'all_patients'], required=True,
                        help='Type of heatmap to generate')
    parser.add_argument('--year', type=str, default='1',
                        help='Year(s) to generate (e.g., "1" or "1-6")')

    args = parser.parse_args()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Parse year range
    if '-' in args.year:
        start, end = map(int, args.year.split('-'))
        years = range(start, end + 1)
    else:
        years = [int(args.year)]

    # Generate heatmaps
    for year in years:
        generate_heatmap(args.patient_type, year, parent_dir)
