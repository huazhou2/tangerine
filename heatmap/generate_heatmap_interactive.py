"""TANGERINE Interactive Heatmap Generator
Creates an HTML dashboard with interactive year and patient type selectors.

Usage:
  python generate_heatmap_interactive.py
"""
import pandas as pd
import numpy as np
import json
import os
import argparse
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


def create_heatmap_data(patient_ids, all_months_list, data_source):
    """Create heatmap arrays: predictions, LRADS, diagnosis markers."""
    heatmap_pred = []
    heatmap_lrads = []
    diagnosis_month_marker = []
    patient_status = []

    for pat_id in patient_ids:
        patient_scans = data_source[data_source['pat_id'] == pat_id].copy()
        patient_scans = patient_scans.sort_values('ct_date')
        diagnosis_date = patient_scans['first_lung_ca_date'].iloc[0]
        is_cancer = patient_scans['cancer_pred'].iloc[0] == 1

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

    return (np.array(heatmap_pred), np.array(heatmap_lrads),
            np.array(diagnosis_month_marker, dtype=bool), patient_status)


def generate_all_heatmap_data(parent_dir):
    """Generate heatmap data for all year/patient_type combinations."""
    data_multi = load_data(parent_dir)
    data_multi['year_month'] = data_multi['ct_date'].dt.to_period('M')

    # Get time range (same for all)
    min_date = data_multi['ct_date'].min()
    max_date = data_multi['ct_date'].max()
    max_diagnosis = data_multi['first_lung_ca_date'].max()
    if pd.notna(max_diagnosis) and max_diagnosis > max_date:
        max_date = max_diagnosis
    all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')
    month_labels = [str(m) for m in all_months]

    # Get all patient lists
    all_patients_sorted = sorted(data_multi['pat_id'].unique())
    cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 1])
    non_cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 0])

    heatmap_data = {}

    # Generate for all years and patient types
    for year in range(1, 7):
        print(f"Generating data for Year {year}...")
        data_multi['pred'] = data_multi[f'pred_{year}_pred']

        # Cancer-only
        cancer_patients_list = data_multi[data_multi['cancer_pred'] == 1]['pat_id'].unique()
        data_cancer = data_multi[data_multi['pat_id'].isin(cancer_patients_list)].copy()
        patient_ids_cancer = sorted(cancer_patients_list)

        pred_cancer, lrads_cancer, diag_cancer, status_cancer = create_heatmap_data(
            patient_ids_cancer, all_months, data_cancer)

        heatmap_data[f'year_{year}_cancer_only'] = {
            'pred': pred_cancer.tolist(),
            'lrads': lrads_cancer.tolist(),
            'diagnosis': diag_cancer.tolist(),
            'patient_ids': [str(p) for p in patient_ids_cancer],
            'patient_status': status_cancer,
            'month_labels': month_labels,
            'n_cancer': len(patient_ids_cancer),
            'n_non_cancer': 0,
            'n_total': len(patient_ids_cancer),
        }

        # Non-cancer only
        data_non_cancer = data_multi[data_multi['pat_id'].isin(non_cancer_patients)].copy()
        pred_non_cancer, lrads_non_cancer, diag_non_cancer, status_non_cancer = create_heatmap_data(
            non_cancer_patients, all_months, data_non_cancer)

        heatmap_data[f'year_{year}_non_cancer_only'] = {
            'pred': pred_non_cancer.tolist(),
            'lrads': lrads_non_cancer.tolist(),
            'diagnosis': diag_non_cancer.tolist(),
            'patient_ids': [str(p) for p in non_cancer_patients],
            'patient_status': status_non_cancer,
            'month_labels': month_labels,
            'n_cancer': 0,
            'n_non_cancer': len(non_cancer_patients),
            'n_total': len(non_cancer_patients),
        }

        # All patients
        pred_all, lrads_all, diag_all, status_all = create_heatmap_data(
            cancer_patients + non_cancer_patients, all_months, data_multi)

        heatmap_data[f'year_{year}_all_patients'] = {
            'pred': pred_all.tolist(),
            'lrads': lrads_all.tolist(),
            'diagnosis': diag_all.tolist(),
            'patient_ids': [str(p) for p in cancer_patients + non_cancer_patients],
            'patient_status': status_all,
            'month_labels': month_labels,
            'n_cancer': len(cancer_patients),
            'n_non_cancer': len(non_cancer_patients),
            'n_total': len(cancer_patients) + len(non_cancer_patients),
        }

    return heatmap_data


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


def create_html_dashboard(heatmap_data, output_dir):
    """Create interactive HTML dashboard with Plotly."""

    # Convert NaN to None for proper JSON serialization
    heatmap_data = convert_nan_to_none(heatmap_data)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TANGERINE Interactive Heatmaps</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        h1 {
            color: #333;
            margin: 0;
        }
        .subtitle {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .controls {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 30px;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        label {
            font-weight: bold;
            color: #333;
        }
        select, button {
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            background: white;
            transition: all 0.3s;
        }
        select:hover, button:hover {
            border-color: #999;
            background: #f9f9f9;
        }
        select:focus {
            outline: none;
            border-color: #4CAF50;
            box-shadow: 0 0 5px rgba(76, 175, 80, 0.3);
        }
        .button-group {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        button {
            padding: 8px 15px;
            background: #f0f0f0;
            color: #333;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        button:hover {
            background: #e0e0e0;
        }
        button.active {
            background: #2196F3;
            color: white;
            border-color: #2196F3;
        }
        .plot-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        #heatmap {
            width: 100%;
        }
        .info {
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
            margin-bottom: 20px;
            color: #333;
            font-size: 14px;
        }
        .footer {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>TANGERINE Interactive Longitudinal Heatmaps</h1>
        <div class="subtitle">Year-wise cancer probability predictions with LRADS scores</div>
    </div>

    <div class="info">
        <strong>Legend:</strong>
        Cell color = Cancer probability (red=high, green=low) |
        Text numbers = LRADS score |
        <span style="color: purple; font-weight: bold;">*</span> = Diagnosis month
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
            <div class="button-group">
                <button id="btnCancerOnly" class="active">Cancer Only</button>
                <button id="btnNonCancerOnly">Non-Cancer Only</button>
                <button id="btnAllPatients">All Patients</button>
            </div>
        </div>
    </div>

    <div class="plot-container">
        <div id="heatmap"></div>
    </div>

    <div class="footer">
        Generated from TANGERINE dataset | Sybil year-wise predictions with LRADS overlays
    </div>

    <script>
        const heatmapData = """ + json.dumps(heatmap_data) + """;

        let currentYear = 1;
        let currentPatientType = 'cancer_only';

        function updateHeatmap() {
            const dataKey = `year_${currentYear}_${currentPatientType}`;
            const data = heatmapData[dataKey];

            if (!data) {
                console.error('Data not found for:', dataKey);
                return;
            }

            const nPatients = data.patient_ids.length;
            const nMonths = data.month_labels.length;

            // Create annotations for LRADS and diagnosis markers
            const annotations = [];

            for (let i = 0; i < nPatients; i++) {
                for (let j = 0; j < nMonths; j++) {
                    const monthLabel = data.month_labels[j];
                    const patientId = data.patient_ids[i];

                    // LRADS annotation
                    if (data.lrads[i] && data.lrads[i][j] !== null && !isNaN(data.lrads[i][j])) {
                        annotations.push({
                            x: monthLabel,
                            y: patientId,
                            text: `${Math.round(data.lrads[i][j])}`,
                            showarrow: false,
                            font: { size: 11, color: 'black', family: 'monospace', weight: 'bold' },
                            xanchor: 'center',
                            yanchor: 'middle',
                            bgcolor: 'rgba(255, 255, 255, 0.8)',
                            bordercolor: 'black',
                            borderwidth: 0.5,
                            borderpad: 2
                        });
                    }

                    // Diagnosis marker
                    if (data.diagnosis[i] && data.diagnosis[i][j]) {
                        annotations.push({
                            x: monthLabel,
                            y: patientId,
                            text: '*',
                            showarrow: false,
                            font: { size: 28, color: 'purple', family: 'Arial', weight: 'bold' },
                            xanchor: 'center',
                            yanchor: 'middle'
                        });
                    }
                }
            }

            // Create hover text
            const hoverText = data.pred.map((row, i) => {
                return row.map((pred, j) => {
                    let text = `<b>Patient:</b> ${data.patient_ids[i]}<br>`;
                    text += `<b>Month:</b> ${data.month_labels[j]}<br>`;
                    text += `<b>Pred:</b> ${pred === null ? 'N/A' : pred.toFixed(3)}<br>`;
                    if (data.lrads[i] && data.lrads[i][j] !== null && !isNaN(data.lrads[i][j])) {
                        text += `<b>LRADS:</b> ${Math.round(data.lrads[i][j])}<br>`;
                    }
                    if (data.diagnosis[i] && data.diagnosis[i][j]) {
                        text += `<b style="color:purple;">★ DIAGNOSIS MONTH</b>`;
                    }
                    return text;
                });
            });

            // Create trace - IMPORTANT: x and y must match annotation labels
            const trace = {
                z: data.pred,
                x: data.month_labels,  // String labels for x-axis
                y: data.patient_ids,   // String labels for y-axis
                type: 'heatmap',
                colorscale: 'RdYlGn-r',
                zmin: 0,
                zmax: 1,
                hovertemplate: '%{customdata}<extra></extra>',
                customdata: hoverText,
                colorbar: {
                    title: `Year ${currentYear}<br>Cancer<br>Prob`,
                    thickness: 15,
                    len: 0.7,
                }
            };

            // Calculate dynamic height based on number of patients
            const minHeight = 600;
            const heightPerPatient = 25;
            const plotHeight = Math.max(minHeight, nPatients * heightPerPatient);

            const layout = {
                title: {
                    text: `TANGERINE ${currentPatientType === 'cancer_only' ? 'Cancer-Only' : currentPatientType === 'non_cancer_only' ? 'Non-Cancer Only' : 'All Patients'} Predictions (Year ${currentYear})<br>` +
                          `<sub>n=${data.n_total} (Cancer=${data.n_cancer}, Non-Cancer=${data.n_non_cancer})</sub>`,
                    font: { size: 16 }
                },
                xaxis: {
                    title: 'Year-Month',
                    tickangle: 45,
                    tickfont: { size: 11 },
                    type: 'category',  // Explicitly set as category
                    categoryorder: 'array',
                    categoryarray: data.month_labels
                },
                yaxis: {
                    title: 'Patient ID',
                    tickfont: { size: 10 },
                    type: 'category',  // Explicitly set as category
                    categoryorder: 'array',
                    categoryarray: data.patient_ids,
                    autorange: 'reversed'
                },
                height: plotHeight,
                margin: { l: 120, b: 120, t: 140, r: 100 },
                hovermode: 'closest',
                annotations: annotations
            };

            Plotly.newPlot('heatmap', [trace], layout, {responsive: true, toImageButtonOptions: {format: 'png', width: 1400, height: plotHeight}});
        }

        // Event listeners
        document.getElementById('yearSelect').addEventListener('change', (e) => {
            currentYear = parseInt(e.target.value);
            updateHeatmap();
        });

        document.getElementById('btnCancerOnly').addEventListener('click', () => {
            currentPatientType = 'cancer_only';
            document.getElementById('btnCancerOnly').classList.add('active');
            document.getElementById('btnNonCancerOnly').classList.remove('active');
            document.getElementById('btnAllPatients').classList.remove('active');
            updateHeatmap();
        });

        document.getElementById('btnNonCancerOnly').addEventListener('click', () => {
            currentPatientType = 'non_cancer_only';
            document.getElementById('btnNonCancerOnly').classList.add('active');
            document.getElementById('btnCancerOnly').classList.remove('active');
            document.getElementById('btnAllPatients').classList.remove('active');
            updateHeatmap();
        });

        document.getElementById('btnAllPatients').addEventListener('click', () => {
            currentPatientType = 'all_patients';
            document.getElementById('btnAllPatients').classList.add('active');
            document.getElementById('btnCancerOnly').classList.remove('active');
            document.getElementById('btnNonCancerOnly').classList.remove('active');
            updateHeatmap();
        });

        // Initialize
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
    parser = argparse.ArgumentParser(description='Generate interactive TANGERINE heatmap dashboard')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for HTML file')

    args = parser.parse_args()

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.output_dir is None:
        args.output_dir = os.path.join(parent_dir, 'results_20260605/heatmaps')

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating heatmap data for all year/patient type combinations...")
    heatmap_data = generate_all_heatmap_data(parent_dir)

    print("Creating interactive HTML dashboard...")
    output_path = create_html_dashboard(heatmap_data, args.output_dir)

    print(f"\n✅ Interactive dashboard created!")
    print(f"📊 Open in browser: {output_path}")
