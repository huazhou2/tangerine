"""TANGERINE Interactive Heatmap Generator - Simplified Version
Creates an HTML dashboard with interactive year and patient type selectors.

Usage:
  python generate_heatmap_interactive.py
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path


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


def create_heatmap_data(patient_ids, all_months_list, data_source, patient_id_map=None):
    """Create heatmap arrays: predictions, LRADS, diagnosis markers."""
    heatmap_pred = []
    heatmap_lrads = []
    diagnosis_month_marker = []
    patient_status = []
    display_ids = []

    for pat_id in patient_ids:
        patient_scans = data_source[data_source['pat_id'] == pat_id].copy()
        patient_scans = patient_scans.sort_values('ct_date')
        diagnosis_date = patient_scans['first_lung_ca_date'].iloc[0]
        is_cancer = patient_scans['cancer_pred'].iloc[0] == 1

        display_id = str(patient_id_map.get(pat_id, pat_id)) if patient_id_map else str(pat_id)

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
        patient_status.append('cancer' if is_cancer else 'non-cancer')
        display_ids.append(display_id)

    return (np.array(heatmap_pred), np.array(heatmap_lrads),
            np.array(diagnosis_month_marker, dtype=bool), patient_status, display_ids)


def convert_nan_to_none(data):
    """Convert NaN values to None for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_nan_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_none(v) for v in data]
    elif isinstance(data, float):
        return None if np.isnan(data) else data
    else:
        return data


def generate_all_heatmap_data(parent_dir):
    """Generate heatmap data for all year/patient_type combinations."""
    data_multi = load_data(parent_dir)
    data_multi['year_month'] = data_multi['ct_date'].dt.to_period('M')

    # Create mapping from pat_id to PatientID for display
    patient_id_map = dict(zip(data_multi['pat_id'], data_multi['PatientID']))

    # Get time range
    min_date = data_multi['ct_date'].min()
    max_date = data_multi['ct_date'].max()
    max_diagnosis = data_multi['first_lung_ca_date'].max()
    if pd.notna(max_diagnosis) and max_diagnosis > max_date:
        max_date = max_diagnosis
    all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')
    month_labels = [str(m) for m in all_months]

    # Get patient lists
    all_patients_sorted = sorted(data_multi['pat_id'].unique())
    cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 1])
    non_cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 0])

    heatmap_data = {}

    for year in range(1, 7):
        print(f"Generating data for Year {year}...")
        data_multi['pred'] = data_multi[f'pred_{year}_pred']

        # Cancer-only
        cancer_patients_list = data_multi[data_multi['cancer_pred'] == 1]['pat_id'].unique()
        data_cancer = data_multi[data_multi['pat_id'].isin(cancer_patients_list)].copy()
        patient_ids_cancer = sorted(cancer_patients_list)

        pred_cancer, lrads_cancer, diag_cancer, status_cancer, display_ids_cancer = create_heatmap_data(
            patient_ids_cancer, all_months, data_cancer, patient_id_map)

        heatmap_data[f'year_{year}_cancer_only'] = {
            'pred': pred_cancer.tolist(),
            'lrads': lrads_cancer.tolist(),
            'diagnosis': diag_cancer.tolist(),
            'patient_ids': display_ids_cancer,
            'patient_status': status_cancer,
            'month_labels': month_labels,
            'n_cancer': len(patient_ids_cancer),
            'n_non_cancer': 0,
            'n_total': len(patient_ids_cancer),
        }

        # Non-cancer only
        data_non_cancer = data_multi[data_multi['pat_id'].isin(non_cancer_patients)].copy()
        pred_non_cancer, lrads_non_cancer, diag_non_cancer, status_non_cancer, display_ids_non_cancer = create_heatmap_data(
            non_cancer_patients, all_months, data_non_cancer, patient_id_map)

        heatmap_data[f'year_{year}_non_cancer_only'] = {
            'pred': pred_non_cancer.tolist(),
            'lrads': lrads_non_cancer.tolist(),
            'diagnosis': diag_non_cancer.tolist(),
            'patient_ids': display_ids_non_cancer,
            'patient_status': status_non_cancer,
            'month_labels': month_labels,
            'n_cancer': 0,
            'n_non_cancer': len(non_cancer_patients),
            'n_total': len(non_cancer_patients),
        }

        # All patients
        pred_all, lrads_all, diag_all, status_all, display_ids_all = create_heatmap_data(
            cancer_patients + non_cancer_patients, all_months, data_multi, patient_id_map)

        heatmap_data[f'year_{year}_all_patients'] = {
            'pred': pred_all.tolist(),
            'lrads': lrads_all.tolist(),
            'diagnosis': diag_all.tolist(),
            'patient_ids': display_ids_all,
            'patient_status': status_all,
            'month_labels': month_labels,
            'n_cancer': len(cancer_patients),
            'n_non_cancer': len(non_cancer_patients),
            'n_total': len(cancer_patients) + len(non_cancer_patients),
        }

    return heatmap_data


def create_html_dashboard(heatmap_data, output_dir):
    """Create interactive HTML dashboard with Plotly."""
    heatmap_data = convert_nan_to_none(heatmap_data)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TANGERINE Interactive Heatmaps</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; }
        h1 { color: #333; margin: 0; }
        .subtitle { color: #666; font-size: 14px; margin-top: 5px; }
        .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 30px; flex-wrap: wrap; align-items: center; }
        .control-group { display: flex; align-items: center; gap: 10px; }
        label { font-weight: bold; color: #333; }
        select, button { padding: 8px 15px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px;
                        cursor: pointer; background: white; }
        button { background: #f0f0f0; color: #333; }
        button.active { background: #2196F3; color: white; border-color: #2196F3; }
        .plot-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        #heatmap { width: 100%; }
        .info { background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; border-radius: 4px;
               margin-bottom: 20px; color: #333; font-size: 14px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TANGERINE Interactive Longitudinal Heatmaps</h1>
        <div class="subtitle">Year-wise cancer probability predictions with LRADS scores</div>
    </div>

    <div class="info">
        <strong>Legend:</strong> Green=low risk, Yellow=medium risk, Red=high risk |
        Numbers=LRADS score | <span style="color:purple;font-weight:bold;">*</span>=Diagnosis month
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="yearSelect">Year:</label>
            <select id="yearSelect">
                <option value="1">Year 1</option>
                <option value="2">Year 2</option>
                <option value="3">Year 3</option>
                <option value="4">Year 4</option>
                <option value="5">Year 5</option>
                <option value="6">Year 6</option>
            </select>
        </div>
        <div class="control-group">
            <label>Patient Type:</label>
            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                <button id="btnCancerOnly" class="active">Cancer Only</button>
                <button id="btnNonCancerOnly">Non-Cancer Only</button>
                <button id="btnAllPatients">All Patients</button>
            </div>
        </div>
    </div>

    <div class="plot-container">
        <div id="heatmap"></div>
    </div>

    <script>
        const heatmapData = """ + json.dumps(heatmap_data) + """;

        let currentYear = 1;
        let currentPatientType = 'cancer_only';

        // RdYlGn_r: Red(high) → Yellow → Green(low)
        const rdylgn_r = [
            [0, 'rgb(0, 104, 157)'],       // Dark green (low)
            [0.25, 'rgb(33, 180, 226)'],   // Light blue-green
            [0.4, 'rgb(171, 217, 233)'],   // Very light blue
            [0.5, 'rgb(254, 224, 144)'],   // Yellow
            [0.6, 'rgb(253, 174, 97)'],    // Orange
            [0.8, 'rgb(215, 48, 39)'],     // Red
            [1, 'rgb(165, 0, 38)']         // Dark red (high)
        ];

        function updateHeatmap() {
            const key = `year_${currentYear}_${currentPatientType}`;
            const data = heatmapData[key];
            if (!data) {
                console.error('No data for:', key);
                return;
            }

            const nPatients = data.patient_ids.length;
            const nMonths = data.month_labels.length;

            // Create hover text
            const hoverText = data.pred.map((row, i) => {
                return row.map((pred, j) => {
                    let text = `<b>Patient:</b> ${data.patient_ids[i]}<br>`;
                    text += `<b>Month:</b> ${data.month_labels[j]}<br>`;
                    text += `<b>Pred:</b> ${pred === null ? 'N/A' : pred.toFixed(3)}<br>`;
                    if (data.lrads[i][j] !== null) {
                        text += `<b>LRADS:</b> ${Math.round(data.lrads[i][j])}<br>`;
                    }
                    if (data.diagnosis[i][j]) {
                        text += `<b style="color:purple;">★ DIAGNOSIS</b>`;
                    }
                    return text;
                });
            });

            // Create annotations
            const annotations = [];
            for (let i = 0; i < nPatients; i++) {
                for (let j = 0; j < nMonths; j++) {
                    // LRADS
                    if (data.lrads[i][j] !== null) {
                        annotations.push({
                            x: data.month_labels[j],
                            y: data.patient_ids[i],
                            text: `${Math.round(data.lrads[i][j])}`,
                            showarrow: false,
                            font: { size: 10, color: 'black', weight: 'bold' },
                            xanchor: 'center', yanchor: 'middle'
                        });
                    }
                    // Diagnosis marker
                    if (data.diagnosis[i][j]) {
                        annotations.push({
                            x: data.month_labels[j],
                            y: data.patient_ids[i],
                            text: '*',
                            showarrow: false,
                            font: { size: 20, color: 'purple', weight: 'bold' },
                            xanchor: 'center', yanchor: 'middle'
                        });
                    }
                }
            }

            const trace = {
                z: data.pred,
                x: data.month_labels,
                y: data.patient_ids,
                type: 'heatmap',
                colorscale: rdylgn_r,
                zmin: 0, zmax: 1,
                hovertemplate: '%{customdata}<extra></extra>',
                customdata: hoverText,
                colorbar: { title: `Year ${currentYear}`, thickness: 15, len: 0.7 }
            };

            const layout = {
                title: {
                    text: `TANGERINE ${currentPatientType === 'cancer_only' ? 'Cancer-Only' : currentPatientType === 'non_cancer_only' ? 'Non-Cancer Only' : 'All Patients'} (Y${currentYear})<br><sub>n=${data.n_total} (Cancer=${data.n_cancer}, Non-Cancer=${data.n_non_cancer})</sub>`,
                    font: { size: 16 }
                },
                xaxis: { title: 'Year-Month', type: 'category', tickangle: 45, tickfont: { size: 11 } },
                yaxis: { title: 'Patient ID', type: 'category', tickfont: { size: 10 } },
                height: Math.max(600, nPatients * 30),
                margin: { l: 120, b: 120, t: 140, r: 100 },
                hovermode: 'closest',
                annotations: annotations
            };

            Plotly.newPlot('heatmap', [trace], layout, {responsive: true});
        }

        document.getElementById('yearSelect').addEventListener('change', (e) => {
            currentYear = parseInt(e.target.value);
            updateHeatmap();
        });

        ['Cancer', 'NonCancer', 'All'].forEach(type => {
            const btn = document.getElementById(`btn${type}Only`);
            if (!btn) return;
            btn.addEventListener('click', () => {
                const typeMap = {'Cancer': 'cancer_only', 'NonCancer': 'non_cancer_only', 'All': 'all_patients'};
                currentPatientType = typeMap[type];
                document.querySelectorAll('.controls button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                updateHeatmap();
            });
        });

        updateHeatmap();
    </script>
</body>
</html>
"""

    output_path = os.path.join(output_dir, 'tangerine_interactive_heatmaps.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    return output_path


if __name__ == '__main__':
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_dir, 'results_20260605/heatmaps')
    os.makedirs(output_dir, exist_ok=True)

    print("Generating heatmap data...")
    heatmap_data = generate_all_heatmap_data(parent_dir)

    print("Creating HTML dashboard...")
    output_path = create_html_dashboard(heatmap_data, output_dir)

    print(f"\n✅ Dashboard created!")
    print(f"📊 Open: {output_path}")
