"""
Generate per-patient PDF reports using Grad-CAM heatmaps for the 6-year survival model.

Reads Grad-CAM volumes from:
  run_dir/gradcam/by_ct_gradcam/<patient_id>/gradcam.nii.gz

Saves PDFs to:
  run_dir/gradcam/reports/

Usage:
    python generate_gradcam_reports.py \\
        --run_dir     outputs/run_XXXX \\
        --meta_csv    /path/to/lungct_with_mrn_anonacc.csv \\
        --images_dir  /path/to/images_3d_swine
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from fpdf import FPDF
from PIL import Image, ImageDraw

NUM_KEY_FRAMES = 9



def top_cam_slices(cam_vol, n=3):
    per_slice = cam_vol.mean(axis=(1, 2))
    top_idx   = np.argsort(per_slice)[::-1][:n]
    return [(int(i), float(per_slice[i])) for i in top_idx]


def select_key_slices(cam_vol, n=NUM_KEY_FRAMES):
    D         = cam_vol.shape[0]
    per_slice = cam_vol.mean(axis=(1, 2))
    valid_idx = [i for i in range(D) if per_slice[i] > 0] or list(range(D))
    lo, hi    = int(len(valid_idx) * 0.1), int(len(valid_idx) * 0.9)
    mid_idx   = valid_idx[lo:hi] or valid_idx
    mid_cam   = [per_slice[i] for i in mid_idx]
    q20, q80  = np.quantile(mid_cam, 0.2), np.quantile(mid_cam, 0.8)
    filtered  = [i for i, a in zip(mid_idx, mid_cam) if q20 <= a <= q80]
    if len(filtered) < n:
        filtered = mid_idx
    step = max(1, len(filtered) // n)
    return [filtered[step * i] for i in range(n)]


def make_slice_overlay(ct_slice, cam_slice, slice_idx, cam_score=None, alpha=0.45):
    size     = 256
    lo, hi   = np.percentile(ct_slice, [1, 99])
    ct_norm  = np.clip((ct_slice - lo) / (hi - lo + 1e-8), 0, 1)
    c_min, c_max = cam_slice.min(), cam_slice.max()
    cam_norm = (cam_slice - c_min) / (c_max - c_min + 1e-8)
    # Resize both to target size before colormap (slices may differ if CT != 256^3)
    ct_norm  = np.array(Image.fromarray((ct_norm  * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)) / 255.0
    cam_norm = np.array(Image.fromarray((cam_norm * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)) / 255.0
    ct_rgb   = (cm.gray(ct_norm)[:, :, :3] * 255).astype(np.uint8)
    cam_rgb  = (cm.jet(cam_norm)[:, :, :3] * 255).astype(np.uint8)
    alpha_map = (cam_norm ** 0.5 * alpha * 1.5)[:, :, np.newaxis]   # alpha ∝ cam; 0 where no activation
    alpha_map = np.clip(alpha_map, 0, alpha)
    blend    = (ct_rgb * (1 - alpha_map) + cam_rgb * alpha_map).astype(np.uint8)
    img      = Image.fromarray(blend)
    draw     = ImageDraw.Draw(img)
    draw.text((2, 2), f'Slice {slice_idx}', fill=(255, 0, 0))
    if cam_score is not None:
        draw.text((2, size - 14), f'Grad-CAM: {cam_score:.4f}', fill=(255, 255, 0))
    return img


def make_key_frame_grid(cam_path, ct_path, n=NUM_KEY_FRAMES, cell=256):
    if not cam_path.exists():
        return None, []
    cam_vol   = sitk.GetArrayFromImage(sitk.ReadImage(str(cam_path)))
    per_slice = cam_vol.mean(axis=(1, 2))
    if ct_path is not None and ct_path.exists():
        ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        while ct_vol.ndim > 3:
            ct_vol = ct_vol.squeeze(0)
        if ct_vol.shape != cam_vol.shape:
            ct_vol = ct_vol[:cam_vol.shape[0], :cam_vol.shape[1], :cam_vol.shape[2]]
    else:
        ct_vol = np.zeros_like(cam_vol)
    key_slices = select_key_slices(cam_vol, n=n)
    top3       = top_cam_slices(cam_vol, n=3)
    num_cols   = 3
    num_rows   = (n + num_cols - 1) // num_cols
    grid       = Image.new('RGB', (num_cols * cell, num_rows * cell))
    for i, s_idx in enumerate(key_slices):
        tile = make_slice_overlay(ct_vol[s_idx], cam_vol[s_idx], s_idx,
                                  cam_score=float(per_slice[s_idx]))
        grid.paste(tile, ((i % num_cols) * cell, (i // num_cols) * cell))
    return grid, top3


class Report(FPDF):
    def header(self):
        pass


def build_report(patient_id, pred_probs_cal, pred_probs_raw, meta, gradcam_dir, images_dir, out_pdf):
    cancer        = int(meta.get('cancer', 0))
    time_at_event = float(meta.get('time_ct_to_last_event_or_followup',
                                    meta.get('time_at_event', 0)))

    if time_at_event < 0:
        cancer_text = ('Already diagnosed with lung cancer on '
                       + str(meta.get('first_lung_ca_date', 'unknown')) + '.')
    elif cancer:
        cancer_text = ('Later diagnosed with lung cancer on '
                       + str(meta.get('first_lung_ca_date', 'unknown')) + '.')
    else:
        cancer_text = ('Not diagnosed with lung cancer till '
                       + str(meta.get('last_enc_date', 'unknown')) + '.')

    yr_lines_cal = '  '.join(f'Y{yr}: {p*100:.1f}%' for yr, p in enumerate(pred_probs_cal, 1))
    yr_lines_raw = '  '.join(f'Y{yr}: {p*100:.1f}%' for yr, p in enumerate(pred_probs_raw, 1))

    body = (
        'TANGERINE (ViT-Large, pretrained on 98k chest CTs via 3D masked autoencoding)\n'
        '6-year lung cancer risk prediction\n\n'
        f'CT series:    {patient_id}\n'
        f'CT date:      {meta.get("ct_date", "N/A")}\n'
        f'Sex:          {meta.get("sex", "N/A")}\n'
        f'Age:          {meta.get("age", "N/A")}\n'
        f'Race:         {meta.get("race", "N/A")}\n\n'
        f'Predicted cumulative cancer probability (calibrated):\n  {yr_lines_cal}\n\n'
        f'Predicted cumulative cancer probability (raw):\n  {yr_lines_raw}\n\n'
        f'Follow-up:    {cancer_text}\n\n'
        'Grad-CAM heatmap (page 2): red = high contribution, blue = low contribution.\n'
        'Attention is masked to lung region; spine and background are excluded.'
    )

    cam_path = gradcam_dir / patient_id / 'gradcam.nii.gz'
    ct_path  = None
    if images_dir is not None:
        for suffix in ['.nii.gz', '_preprocessed.nii.gz']:
            candidate = Path(images_dir) / f'{patient_id}{suffix}'
            if candidate.exists():
                ct_path = candidate
                break

    grid_img, top3 = make_key_frame_grid(cam_path, ct_path)

    pdf = Report()
    pdf.set_auto_page_break(auto=False)

    # ── Page 1: patient info + predictions ───────────────────────────────────
    pdf.add_page()
    pdf.set_xy(0, 5)
    pdf.set_font('Arial', 'B', 20.0)
    pdf.cell(w=210, h=18,
             txt=f'TANGERINE Grad-CAM Report - CT {patient_id}',
             ln=1, border=0, align='C', fill=False)
    pdf.set_font('Arial', '', 12.0)
    pdf.set_xy(10, 28)
    safe_body = body.replace('\u2014', '-').replace('\u2013', '-').encode('latin-1', errors='replace').decode('latin-1')
    pdf.multi_cell(w=190, h=7, txt=safe_body, border=0, align='L', fill=False)

    # ── Page 2: image grid + top-slice summary ────────────────────────────────
    pdf.add_page()
    pdf.set_font('Arial', 'B', 11)
    pdf.set_xy(0, 3)
    pdf.cell(w=210, h=8,
             txt='Grad-CAM - 9 representative axial slices (lung-masked CT + overlay)',
             ln=1, align='C')
    pdf.set_font('Arial', '', 9)
    pdf.cell(w=210, h=5,
             txt='Red label = slice index.  Yellow label = mean Grad-CAM intensity.',
             ln=1, align='C')

    if grid_img is not None:
        tmp_img = str(out_pdf).replace('.pdf', '_tmp_grid.png')
        grid_img.save(tmp_img)
        pdf.image(tmp_img, x=0, y=18, w=210, h=210)
        os.remove(tmp_img)
        if top3:
            top3_lines = '   '.join(
                f'Rank {i+1}: slice {s_idx} (Grad-CAM: {score:.4f})'
                for i, (s_idx, score) in enumerate(top3)
            )
            pdf.set_xy(10, 232)
            pdf.set_font('Arial', '', 9)
            pdf.cell(w=190, h=6, txt='Peak slices to review in PACS:  ' + top3_lines,
                     ln=1, align='L')
    else:
        pdf.set_xy(10, 120)
        pdf.set_font('Arial', '', 12)
        pdf.cell(w=190, h=10, txt='Grad-CAM map not available for this patient.',
                 ln=1, align='C')

    pdf.output(str(out_pdf), 'F')


def main(args):
    run_dir     = Path(args.run_dir)
    gradcam_dir = run_dir / 'attention' / 'grad_cam' / 'by_ct'
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = run_dir / 'test_predictions.csv'
    if not preds_csv.exists():
        raise FileNotFoundError(f'test_predictions.csv not found in {run_dir}')
    preds_df = pd.read_csv(preds_csv)
    preds_df['patient_id'] = preds_df['patient_id'].apply(lambda x: str(int(float(x))))
    print(f'Loaded {len(preds_df)} test predictions')

    meta_df  = None
    meta_key = None
    if args.meta_csv and Path(args.meta_csv).exists():
        meta_df  = pd.read_csv(args.meta_csv)
        meta_key = 'anon_acc' if 'anon_acc' in meta_df.columns else meta_df.columns[0]
        meta_df[meta_key] = meta_df[meta_key].apply(lambda x: str(int(float(x))))
        print(f'Loaded metadata: {len(meta_df)} rows  (key: {meta_key})')
    else:
        print('No metadata CSV — sex/age/race will show as N/A')

    target_df = preds_df if args.all_patients else \
                preds_df[preds_df['cancer'] == 1].reset_index(drop=True)
    print(f'Generating Grad-CAM reports for {len(target_df)} patients '
          f'({"all" if args.all_patients else "cancer=1 only"})')

    for _, row in target_df.iterrows():
        pid = str(row['patient_id'])
        if meta_df is not None:
            meta_rows = meta_df[meta_df[meta_key] == pid]
            meta = meta_rows.iloc[0].to_dict() if len(meta_rows) > 0 else {}
        else:
            meta = {}
        meta.setdefault('cancer',       int(row.get('cancer', 0)))
        meta.setdefault('time_at_event', float(row.get('time_at_event', 0)))

        # Per-year calibrated and raw probabilities
        MAX_YR = 6
        pred_probs_raw = []
        pred_probs_cal = []
        for yr in range(1, MAX_YR + 1):
            raw = float(row.get(f'pred_{yr}', row.get(f'pred_yr{yr}', 0)))
            cal = float(row.get(f'pred_{yr}_calibrated', raw))
            pred_probs_raw.append(raw)
            pred_probs_cal.append(cal)

        out_pdf = out_dir / f'ct_{pid}_gradcam_report.pdf'
        try:
            build_report(
                patient_id=pid,
                pred_probs_cal=pred_probs_cal,
                pred_probs_raw=pred_probs_raw,
                meta=meta,
                gradcam_dir=gradcam_dir,
                images_dir=args.images_dir,
                out_pdf=out_pdf,
            )
            print(f'  OK  {pid}  ->  {out_pdf.name}')
        except Exception as e:
            print(f'  ERR {pid}  ->  {e}')

    print(f'\nDone. Grad-CAM reports saved to: {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir',      required=True)
    p.add_argument('--meta_csv',     default=None)
    p.add_argument('--images_dir',   default=None)
    p.add_argument('--output_dir',   default=None)
    p.add_argument('--all_patients', action='store_true', default=False)
    args = p.parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.run_dir) / 'attention' / 'grad_cam' / 'reports')
    main(args)
