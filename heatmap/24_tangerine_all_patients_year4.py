"""TANGERINE All Patients Year 4 (Cancer + Non-Cancer)"""
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

data_multi['year_month'] = data_multi['ct_date'].dt.to_period('M')
data_multi['pred'] = data_multi['pred_4_pred']

# Sort: cancer patients first (sorted), then non-cancer (sorted)
all_patients_sorted = sorted(data_multi['pat_id'].unique())
cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 1])
non_cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 0])
all_patients = cancer_patients + non_cancer_patients

min_date = data_multi['ct_date'].min()
max_date = data_multi['ct_date'].max()
max_diagnosis = data_multi['first_lung_ca_date'].max()
if pd.notna(max_diagnosis) and max_diagnosis > max_date:
    max_date = max_diagnosis

all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')

print("="*80)
print("TANGERINE ALL PATIENTS YEAR 4 (CANCER + NON-CANCER)")
print("="*80)
print(f"Cancer patients: {len(cancer_patients)}")
print(f"Non-cancer patients: {len(non_cancer_patients)}")
print(f"Total: {len(all_patients)}")

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

pred_all, lrads_all, diagnosis_all = create_heatmap_data(all_patients, all_months, data_multi)

fig, ax = plt.subplots(figsize=(60, 333))
im = ax.imshow(pred_all, aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

for i in range(len(all_patients)):
    for j in range(len(all_months)):
        lrads_val = lrads_all[i, j]
        if not np.isnan(lrads_val):
            ax.text(j, i, f'{int(lrads_val)}', color='black', fontsize=28, ha='center', va='center',
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1),
                    zorder=8)
        if diagnosis_all[i, j]:
            ax.text(j, i, '*', color='purple', fontsize=70, ha='center', va='center',
                    fontweight='bold', zorder=10)

# Add separator line between cancer and non-cancer
if len(cancer_patients) > 0:
    ax.axhline(len(cancer_patients) - 0.5, color='black', linewidth=2, linestyle='--', zorder=5)

ax.set_xticks(np.arange(0, len(all_months), 6))
ax.set_xticklabels([month_labels[i] for i in np.arange(0, len(all_months), 6)], rotation=45, ha='right', fontsize=16)
ax.set_yticks(np.arange(len(all_patients)))
y_colors = ['red'] * len(cancer_patients) + ['black'] * len(non_cancer_patients)
y_labels = [str(p) for p in all_patients]
ax.set_yticklabels(y_labels, fontsize=16)
for tick, color in zip(ax.get_yticklabels(), y_colors):
    tick.set_color(color)
    tick.set_fontweight('bold')

ax.set_xlabel('Year-Month', fontsize=18, fontweight='bold')
ax.set_ylabel('Patient ID', fontsize=13, fontweight='bold')

title = f'TANGERINE All Patients Year 4: n={len(all_patients)} (Cancer={len(cancer_patients)}, Non-Cancer={len(non_cancer_patients)})\n'
title += f'Cell color = Year-4 cancer probability | PURPLE * = Diagnosis month'
ax.set_title(title, fontsize=32, fontweight='bold', pad=15)

for i in range(len(all_months)+1):
    ax.axvline(i-0.5, color='gray', linewidth=0.5, alpha=0.2)
rect = Rectangle((-0.5, -0.5), len(all_months), len(all_patients),
                linewidth=3, edgecolor='black', facecolor='none', zorder=5)
ax.add_patch(rect)

cbar = plt.colorbar(im, ax=ax, label=f'Year-4 Cancer Probability')

plt.tight_layout()
heatmap_dir = os.path.join(parent_dir, 'results_20260605/heatmaps')
os.makedirs(heatmap_dir, exist_ok=True)
plt.savefig(os.path.join(heatmap_dir, 'tangerine_all_patients_year4.pdf'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: tangerine_all_patients_year4.pdf")
plt.close()

print(f"✅ Year 4 complete!")
