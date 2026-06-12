"""TANGERINE Interactive Heatmap Generator
Generates interactive HTML heatmaps matching the matplotlib version exactly.
"""
import pandas as pd
import numpy as np
import json
import os


def load_data(parent_dir):
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


def create_heatmap_data(patient_ids, all_months_list, data_source, patient_id_map):
    heatmap_pred = []
    heatmap_lrads = []
    diagnosis_month_marker = []
    display_ids = []

    for pat_id in patient_ids:
        patient_scans = data_source[data_source['pat_id'] == pat_id].copy()
        patient_scans = patient_scans.sort_values('ct_date')
        diagnosis_date = patient_scans['first_lung_ca_date'].iloc[0]

        display_id = str(patient_id_map.get(pat_id, pat_id))

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
        display_ids.append(display_id)

    return np.array(heatmap_pred), np.array(heatmap_lrads), np.array(diagnosis_month_marker, dtype=bool), display_ids


def generate_all_heatmap_data(parent_dir):
    data_multi = load_data(parent_dir)
    data_multi['year_month'] = data_multi['ct_date'].dt.to_period('M')
    patient_id_map = dict(zip(data_multi['pat_id'], data_multi['PatientID']))

    min_date = data_multi['ct_date'].min()
    max_date = data_multi['ct_date'].max()
    max_diagnosis = data_multi['first_lung_ca_date'].max()
    if pd.notna(max_diagnosis) and max_diagnosis > max_date:
        max_date = max_diagnosis
    all_months = pd.period_range(start=min_date.to_period('M'), end=max_date.to_period('M'), freq='M')
    month_labels = [str(m) for m in all_months]

    all_patients_sorted = sorted(data_multi['pat_id'].unique())
    cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 1])
    non_cancer_patients = sorted([p for p in all_patients_sorted if data_multi[data_multi['pat_id']==p]['cancer_pred'].iloc[0] == 0])

    heatmap_data = {}

    # Get consistent cancer patient list for all views
    cancer_patients_list = sorted(data_multi[data_multi['cancer_pred'] == 1]['pat_id'].unique())
    non_cancer_patients_list = sorted(data_multi[data_multi['cancer_pred'] == 0]['pat_id'].unique())

    for year in range(1, 7):
        print(f"Generating Year {year}...")
        data_multi['pred'] = data_multi[f'pred_{year}_pred']

        # Cancer-only
        data_cancer = data_multi[data_multi['pat_id'].isin(cancer_patients_list)].copy()
        pred_cancer, lrads_cancer, diag_cancer, display_ids_cancer = create_heatmap_data(
            cancer_patients_list, all_months, data_cancer, patient_id_map)

        heatmap_data[f'year_{year}_cancer_only'] = {
            'pred': [[float(v) if not np.isnan(v) else None for v in row] for row in pred_cancer],
            'lrads': [[float(v) if not np.isnan(v) else None for v in row] for row in lrads_cancer],
            'diagnosis': diag_cancer.tolist(),
            'patient_ids': display_ids_cancer,
            'month_labels': month_labels,
            'n_cancer': len(cancer_patients_list),
            'n_non_cancer': 0,
            'n_total': len(cancer_patients_list),
        }

        # Non-cancer only
        data_non_cancer = data_multi[data_multi['pat_id'].isin(non_cancer_patients_list)].copy()
        pred_non_cancer, lrads_non_cancer, diag_non_cancer, display_ids_non_cancer = create_heatmap_data(
            non_cancer_patients_list, all_months, data_non_cancer, patient_id_map)

        heatmap_data[f'year_{year}_non_cancer_only'] = {
            'pred': [[float(v) if not np.isnan(v) else None for v in row] for row in pred_non_cancer],
            'lrads': [[float(v) if not np.isnan(v) else None for v in row] for row in lrads_non_cancer],
            'diagnosis': diag_non_cancer.tolist(),
            'patient_ids': display_ids_non_cancer,
            'month_labels': month_labels,
            'n_cancer': 0,
            'n_non_cancer': len(non_cancer_patients_list),
            'n_total': len(non_cancer_patients_list),
        }

        # All patients - cancer on top, then non-cancer
        pred_all, lrads_all, diag_all, display_ids_all = create_heatmap_data(
            cancer_patients_list + non_cancer_patients_list, all_months, data_multi, patient_id_map)

        heatmap_data[f'year_{year}_all_patients'] = {
            'pred': [[float(v) if not np.isnan(v) else None for v in row] for row in pred_all],
            'lrads': [[float(v) if not np.isnan(v) else None for v in row] for row in lrads_all],
            'diagnosis': diag_all.tolist(),
            'patient_ids': display_ids_all,
            'month_labels': month_labels,
            'n_cancer': len(cancer_patients_list),
            'n_non_cancer': len(non_cancer_patients_list),
            'n_total': len(cancer_patients_list) + len(non_cancer_patients_list),
        }

    return heatmap_data


def create_html_dashboard(heatmap_data, output_dir):
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TANGERINE Interactive Heatmaps</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { text-align: center; margin-bottom: 30px; }
        h1 { color: #333; margin: 0; }
        .controls { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
        label { font-weight: bold; }
        select, button { padding: 8px 15px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }
        button { background: #f0f0f0; }
        button.active { background: #2196F3; color: white; border-color: #2196F3; }
        .plot-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        #heatmap { width: 100%; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TANGERINE Interactive Longitudinal Heatmaps</h1>
    </div>

    <div class="controls">
        <label>Year: <select id="yearSelect">
            <option value="1">Year 1</option>
            <option value="2">Year 2</option>
            <option value="3">Year 3</option>
            <option value="4">Year 4</option>
            <option value="5">Year 5</option>
            <option value="6">Year 6</option>
        </select></label>

        <label>Patient Type:</label>
        <button id="btn1" class="active">Cancer Only</button>
        <button id="btn2">Non-Cancer Only</button>
        <button id="btn3">All Patients</button>
    </div>

    <div class="plot-container">
        <div id="heatmap"></div>
    </div>

    <script>
        const data = """ + json.dumps(heatmap_data) + """;

        // RdYlGn_r colorscale: Green(0) -> Yellow(0.5) -> Red(1)
        const rdylgn_r = [
            [0.0, 'rgb(0, 104, 55)'],
            [0.25, 'rgb(134, 203, 102)'],
            [0.5, 'rgb(254, 254, 189)'],
            [0.75, 'rgb(248, 139, 81)'],
            [1.0, 'rgb(165, 0, 38)']
        ];

        let year = '1';
        let ptype = 'cancer_only';

        function plot() {
            const key = `year_${year}_${ptype}`;
            const d = data[key];

            if (!d) {
                console.log('No data for', key);
                return;
            }

            console.log(`Plotting ${key}: ${d.patient_ids.length} patients, ${d.month_labels.length} months`);

            // Create hover text with LRADS and diagnosis info
            const hoverText = [];
            const customData = [];
            for (let i = 0; i < d.patient_ids.length; i++) {
                const row = [];
                const customRow = [];
                for (let j = 0; j < d.month_labels.length; j++) {
                    let text = `Patient: ${d.patient_ids[i]}<br>Month: ${d.month_labels[j]}<br>Pred: ${d.pred[i][j] !== null ? d.pred[i][j].toFixed(3) : 'N/A'}`;

                    if (d.lrads[i][j] !== null) {
                        text += `<br>LRADS: ${Math.round(d.lrads[i][j])}`;
                    }
                    if (d.diagnosis[i][j]) {
                        text += `<br><b style="color:purple;">★ DIAGNOSIS MONTH</b>`;
                    }
                    row.push(text);
                    customRow.push(d.pred[i][j]);
                }
                hoverText.push(row);
                customData.push(customRow);
            }

            const trace = {
                z: d.pred,
                x: d.month_labels,
                y: d.patient_ids,
                type: 'heatmap',
                colorscale: rdylgn_r,
                zmin: 0,
                zmax: 1,
                hovertemplate: '%{customdata.text}<extra></extra>',
                customdata: d.pred.map((row, i) => row.map((val, j) => ({
                    text: (() => {
                        let text = `Patient: ${d.patient_ids[i]}<br>Month: ${d.month_labels[j]}<br>Pred: ${val !== null ? val.toFixed(3) : 'N/A'}`;
                        if (d.lrads[i][j] !== null) {
                            text += `<br>LRADS: ${Math.round(d.lrads[i][j])}`;
                        }
                        if (d.diagnosis[i][j]) {
                            text += `<br><b style="color:purple;">★ DIAGNOSIS</b>`;
                        }
                        return text;
                    })()
                }))),
                colorbar: { title: `Y${year}` }
            };

            // Create overlay traces for LRADS (text) and diagnosis (*)
            const lradsX = [], lradsY = [], lradsText = [];
            const diagnosisX = [], diagnosisY = [], diagnosisText = [];

            for (let i = 0; i < d.patient_ids.length; i++) {
                for (let j = 0; j < d.month_labels.length; j++) {
                    // LRADS overlay
                    if (d.lrads[i][j] !== null) {
                        lradsX.push(d.month_labels[j]);
                        lradsY.push(d.patient_ids[i]);
                        lradsText.push(String(Math.round(d.lrads[i][j])));
                    }
                    // Diagnosis overlay
                    if (d.diagnosis[i][j]) {
                        diagnosisX.push(d.month_labels[j]);
                        diagnosisY.push(d.patient_ids[i]);
                        diagnosisText.push('*');
                    }
                }
            }

            const lradsTrace = {
                x: lradsX,
                y: lradsY,
                text: lradsText,
                type: 'scatter',
                mode: 'text',
                textfont: { size: 11, color: 'black', weight: 'bold' },
                hoverinfo: 'skip',
                showlegend: false
            };

            const diagnosisTrace = {
                x: diagnosisX,
                y: diagnosisY,
                text: diagnosisText,
                type: 'scatter',
                mode: 'text',
                textfont: { size: 22, color: 'purple', weight: 'bold' },
                hoverinfo: 'skip',
                showlegend: false
            };

            // Add separator line for all-patients (between cancer and non-cancer)
            const annotations = [];
            if (ptype === 'all_patients' && d.n_cancer > 0) {
                annotations.push({
                    type: 'line',
                    x0: -0.5, x1: d.month_labels.length - 0.5,
                    y0: d.n_cancer - 0.5, y1: d.n_cancer - 0.5,
                    line: { color: 'black', width: 2, dash: 'dash' }
                });
            }

            // Calculate height based on patient count - minimal padding
            let height = 600;
            if (d.patient_ids.length <= 12) {
                height = 600 + d.patient_ids.length * 20;  // Cancer-only: 20px per row
            } else if (d.patient_ids.length <= 50) {
                height = 50 + d.patient_ids.length * 14;  // Non-cancer: 14px per row, minimal padding
            } else {
                height = 80 + d.patient_ids.length * 11;  // All-patients: 11px per row, minimal padding
            }

            // Use much smaller margins for larger plots
            let topMargin = 25;
            let bottomMargin = 15;
            if (d.patient_ids.length > 50) {
                topMargin = 3;   // All-patients: extremely tight
                bottomMargin = 10;
            } else if (d.patient_ids.length > 12) {
                topMargin = 5;   // Non-cancer: very tight
                bottomMargin = 10;
            }

            const layout = {
                title: {
                    text: `TANGERINE ${ptype === 'cancer_only' ? 'Cancer-Only' : ptype === 'non_cancer_only' ? 'Non-Cancer Only' : 'All Patients'} Year ${year}`,
                    y: 0.8,
                    yanchor: 'top'
                },
                xaxis: { title: 'Month', type: 'category' },
                yaxis: {
                    title: 'Patient ID',
                    type: 'category',
                    categoryorder: 'array',
                    categoryarray: d.patient_ids,
                    autorange: 'reversed'
                },
                height: height,
                margin: { l: 120, b: bottomMargin, t: topMargin, r: 80 },
                hovermode: 'closest',
                shapes: annotations
            };

            Plotly.newPlot('heatmap', [trace, lradsTrace, diagnosisTrace], layout, {responsive: true});
        }

        document.getElementById('yearSelect').addEventListener('change', e => {
            year = e.target.value;
            plot();
        });

        document.getElementById('btn1').addEventListener('click', () => {
            ptype = 'cancer_only';
            document.getElementById('btn1').classList.add('active');
            document.getElementById('btn2').classList.remove('active');
            document.getElementById('btn3').classList.remove('active');
            plot();
        });

        document.getElementById('btn2').addEventListener('click', () => {
            ptype = 'non_cancer_only';
            document.getElementById('btn2').classList.add('active');
            document.getElementById('btn1').classList.remove('active');
            document.getElementById('btn3').classList.remove('active');
            plot();
        });

        document.getElementById('btn3').addEventListener('click', () => {
            ptype = 'all_patients';
            document.getElementById('btn3').classList.add('active');
            document.getElementById('btn1').classList.remove('active');
            document.getElementById('btn2').classList.remove('active');
            plot();
        });

        plot();
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

    print("Generating data...")
    heatmap_data = generate_all_heatmap_data(parent_dir)

    print("Creating HTML...")
    output_path = create_html_dashboard(heatmap_data, output_dir)

    print(f"✅ Created: {output_path}")
