"""TANGERINE Cancer-Only Year 1 Prediction"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

cancer_patients_list = data_multi[data_multi['cancer_pred'] == 1]['pat_id'].unique()
data_cancer = data_multi[data_multi['pat_id'].isin(cancer_patients_list)].copy()

data_cancer['year_month'] = data_cancer['ct_date'].dt.to_period('M')
data_cancer['pred'] = data_cancer['pred_1_pred']

cancer_patients = sorted(cancer_patients_list)

min_date = data_cancer['ct_date'].min()
max_date = data_cancer['ct_date'].max()
max_diagnosis = data_cancer['first_lung_ca_date'].max()
if pd.notna(max_diagnosis) and max_diagnosis > max_date:
    max_date = max_diagnosis

all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')

print("="*80)
print("TANGERINE CANCER-ONLY LONGITUDINAL HEATMAP — YEAR 1")
print("="*80)
print(f"\nCancer patients: {len(cancer_patients)}")
print(f"Total scans: {len(data_cancer)}")
print(f"Scans with LRADS: {data_cancer['lrads'].notna().sum()}")
print(f"Time range: {all_months[0]} to {all_months[-1]} ({len(all_months)} months)")

def create_heatmap_data(patient_ids, all_months_list, data_source):
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

month_labels = [str(m) for m in all_months]
cmap = plt.cm.RdYlGn_r

pred_cancer, lrads_cancer, diagnosis_cancer = create_heatmap_data(cancer_patients, all_months, data_cancer)

fig, ax = plt.subplots(figsize=(60, 12))
im = ax.imshow(pred_cancer, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

for i in range(len(cancer_patients)):
    for j in range(len(all_months)):
        lrads_val = lrads_cancer[i, j]
        if not np.isnan(lrads_val):
            ax.text(j, i, f'{int(lrads_val)}', color='black', fontsize=28, ha='center', va='center',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1),
                    zorder=8)
        if diagnosis_cancer[i, j]:
            ax.text(j, i, '*', color='purple', fontsize=28, ha='center', va='center',
                    fontweight='bold', zorder=10)

ax.set_xticks(np.arange(0, len(all_months), 6))
ax.set_xticklabels([month_labels[i] for i in np.arange(0, len(all_months), 6)], rotation=45, ha='right', fontsize=16)
ax.set_yticks(np.arange(len(cancer_patients)))
ax.set_yticklabels([str(p) for p in cancer_patients], fontsize=16, color='red', fontweight='bold')

ax.set_xlabel('Year-Month', fontsize=18, fontweight='bold')
ax.set_ylabel('Patient ID (Cancer Diagnosed)', fontsize=13, fontweight='bold')

title = f'TANGERINE Cancer-Only Predictions (Year 1): n={len(cancer_patients)} patients\n'
title += f'Cell color = Year-1 cancer probability | PURPLE * = Diagnosis month'
ax.set_title(title, fontsize=32, fontweight='bold', pad=15)

for i in range(len(all_months)+1):
    ax.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.2)
rect = Rectangle((-0.5, -0.5), len(all_months), len(cancer_patients),
                linewidth=3, edgecolor='black', facecolor='none', zorder=5)
ax.add_patch(rect)

cbar = plt.colorbar(im, ax=ax, label='Year-1 Cancer Probability')

plt.tight_layout()
heatmap_dir = os.path.join(parent_dir, 'results_20260605/heatmaps')
os.makedirs(heatmap_dir, exist_ok=True)
plt.savefig(os.path.join(heatmap_dir, 'tangerine_cancer_only_year1.pdf'), dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: tangerine_cancer_only_year1.pdf")
plt.close()

print(f"✅ Year 1 complete!")
