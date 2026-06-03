"""
Generate per-patient PDF reports for TANGERINE 6-year survival predictions,
following the same format as Sybil's get_pred.py reports.

For each cancer patient (or all patients with --all_patients):
  Page 1 — text: patient metadata, 6-year risk predictions, follow-up outcome
  Page 2 — image: 3x3 grid of 9 key axial slices with attention overlay
             (key frames selected with same logic as Sybil's GIF frame filter)

Usage:
    python generate_patient_reports.py \
        --run_dir       outputs/run_20260309_125558 \
        --meta_csv      /path/to/lungct_with_mrn_anonacc.csv \
        --images_dir    /path/to/images_3d_swine \
        --output_dir    outputs/run_20260309_125558/reports
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from fpdf import FPDF
from PIL import Image, ImageDraw

NUM_KEY_FRAMES = 9   # 3x3 grid, same as Sybil




def top_attention_slices(attn_vol, n=3):
    """Return top-n axial slice indices sorted by mean attention, with scores."""
    per_slice = attn_vol.mean(axis=(1, 2))
    top_idx   = np.argsort(per_slice)[::-1][:n]
    return [(int(i), float(per_slice[i])) for i in top_idx]


# ── Slice selection (mirrors Sybil's GIF frame filter logic) ──────────────────

def select_key_slices(attn_vol, n=NUM_KEY_FRAMES):
    """
    Given a 3D attention volume [D, H, W], return n representative axial
    slice indices using Sybil's filtering strategy:
      1. Compute per-slice attention (mean over H×W).
      2. Keep middle 80% of depth indices (drop top and bottom 10%).
      3. Within that range, keep slices where attention falls between the
         20th and 80th percentile (avoids flat/noisy extremes).
      4. Return n evenly-spaced indices from the filtered set.
    """
    D = attn_vol.shape[0]
    per_slice = attn_vol.mean(axis=(1, 2))   # [D]

    # Sybil step 1: indices where attention is non-zero (equivalent to diff check)
    valid_idx = [i for i in range(D) if per_slice[i] > 0]
    if len(valid_idx) == 0:
        valid_idx = list(range(D))

    # Sybil step 2: keep middle 80%
    lo = int(len(valid_idx) * 0.1)
    hi = int(len(valid_idx) * 0.9)
    mid_idx = valid_idx[lo:hi]
    if len(mid_idx) == 0:
        mid_idx = valid_idx

    # Sybil step 3: attention between 20th and 80th percentile
    mid_attn = [per_slice[i] for i in mid_idx]
    q20 = np.quantile(mid_attn, 0.2)
    q80 = np.quantile(mid_attn, 0.8)
    filtered = [i for i, a in zip(mid_idx, mid_attn) if q20 <= a <= q80]
    if len(filtered) < n:
        filtered = mid_idx  # fallback: use all middle slices

    # Sybil step 4: evenly spaced
    step = max(1, len(filtered) // n)
    key = [filtered[step * i] for i in range(n)]
    return key


# ── Overlay image generation ───────────────────────────────────────────────────

def make_slice_overlay(ct_slice, attn_slice, slice_idx, attn_score=None, alpha=0.5):
    """
    Blend a grayscale CT slice (lung window) with a hot-colormap attention heatmap.
    Returns a PIL Image (RGB, 256×256).
    """
    size = 256

    # Percentile-based normalisation — works for both HU and [0,1] pre-normalised CTs
    lo, hi  = np.percentile(ct_slice, [1, 99])
    ct_norm = np.clip((ct_slice - lo) / (hi - lo + 1e-8), 0, 1)

    # Normalise attention to [0, 1] within this slice
    a_min, a_max = attn_slice.min(), attn_slice.max()
    attn_norm = (attn_slice - a_min) / (a_max - a_min + 1e-8)

    # Resize both to target size before colormap (slices may differ if CT != 256^3)
    ct_norm   = np.array(Image.fromarray((ct_norm   * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)) / 255.0
    attn_norm = np.array(Image.fromarray((attn_norm * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)) / 255.0

    # Render CT as grayscale RGB
    ct_rgb = (cm.gray(ct_norm)[:, :, :3] * 255).astype(np.uint8)

    # Render attention as hot colourmap RGB
    attn_rgb = (cm.hot(attn_norm)[:, :, :3] * 255).astype(np.uint8)

    # Adaptive alpha: proportional to attention intensity so low-attention
    # regions (background, outside lung) stay as CT with minimal overlay
    alpha_map = (attn_norm ** 0.5 * alpha * 1.5)[:, :, np.newaxis]
    alpha_map = np.clip(alpha_map, 0, alpha)

    blend = (ct_rgb * (1 - alpha_map) + attn_rgb * alpha_map).astype(np.uint8)
    img = Image.fromarray(blend)

    # Labels: slice index top-left, attention score bottom-left (matching Sybil style)
    draw = ImageDraw.Draw(img)
    draw.text((2, 2), f'Slice {slice_idx}', fill=(255, 0, 0))
    if attn_score is not None:
        draw.text((2, size - 14), f'Attn: {attn_score:.4f}', fill=(255, 255, 0))
    return img


def make_key_frame_grid(attn_path, ct_path, n=NUM_KEY_FRAMES, cell=256):
    """
    Build an (n_rows * cell) x (n_cols * cell) RGB PIL Image
    showing n key axial slices with attention overlay in a 3x3 grid.
    Returns (grid_img, top3_slices) or (None, []) if attention file is missing.
    top3_slices: list of (slice_idx, mean_attn_score) for top-3 attention slices.
    """
    if not attn_path.exists():
        return None, []

    attn_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(attn_path)))  # [D,H,W]
    per_slice = attn_vol.mean(axis=(1, 2))

    # Load CT if available, else use zeros
    if ct_path is not None and ct_path.exists():
        ct_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)
        while ct_vol.ndim > 3:
            ct_vol = ct_vol.squeeze(0)
        if ct_vol.shape != attn_vol.shape:
            ct_vol = ct_vol[:attn_vol.shape[0], :attn_vol.shape[1], :attn_vol.shape[2]]
    else:
        ct_vol = np.zeros_like(attn_vol)

    key_slices = select_key_slices(attn_vol, n=n)
    top3       = top_attention_slices(attn_vol, n=3)

    num_cols = 3
    num_rows = (n + num_cols - 1) // num_cols
    grid_w = num_cols * cell
    grid_h = num_rows * cell
    grid = Image.new('RGB', (grid_w, grid_h))

    for i, s_idx in enumerate(key_slices):
        ct_s       = ct_vol[s_idx]
        at_s       = attn_vol[s_idx]
        attn_score = float(per_slice[s_idx])
        tile = make_slice_overlay(ct_s, at_s, s_idx, attn_score=attn_score)
        col  = i % num_cols
        row  = i // num_cols
        grid.paste(tile, (col * cell, row * cell))

    return grid, top3


# ── PDF report ─────────────────────────────────────────────────────────────────

class Report(FPDF):
    def header(self):
        pass  # no automatic header


def build_report(patient_id, preds, meta, attn_dir, images_dir, out_pdf):
    """
    preds  : dict with keys pred_1..pred_6 (calibrated probabilities)
    meta   : dict / Series with age, sex, race, smoke, ct_date,
                              cancer, time_at_event, first_lung_ca_date, last_enc_date
    """
    cancer         = int(meta.get('cancer', 0))
    time_at_event  = float(meta.get('time_ct_to_last_event_or_followup',
                                     meta.get('time_at_event', 0)))

    # ── Cancer follow-up text (mirrors Sybil exactly) ─────────────────
    if time_at_event < 0:
        cancer_text = ('The patient was already diagnosed with lung cancer on '
                       + str(meta.get('first_lung_ca_date', 'unknown')) + '.')
    elif cancer:
        cancer_text = ('The patient was later diagnosed with lung cancer on '
                       + str(meta.get('first_lung_ca_date', 'unknown')) + '.')
    else:
        cancer_text = ('The patient was not diagnosed with lung cancer till '
                       + str(meta.get('last_enc_date', 'unknown')) + '.')

    # ── Prediction lines ───────────────────────────────────────────────
    pred_lines = '\n'.join(
        f'Year {t}: {preds.get(f"pred_{t}", 0)*100:.2f}%' for t in range(1, 7)
    )

    body = (
        'We used a machine learning model called "TANGERINE" - a Vision Transformer '
        '(ViT-Large) pretrained on 98,000 chest CT scans via 3D masked autoencoding, '
        'equipped with a 6-year survival prediction head - to estimate the cumulative '
        'risk of lung cancer development over the next 6 years for the following case:\n\n'
        f'CT series:         {patient_id}\n'
        f'CT scan date:   {meta.get("ct_date", "N/A")}\n'
        f'Patient sex:      {meta.get("sex", "N/A")}\n'
        f'Patient Age:      {meta.get("age", "N/A")}\n'
        f'Patient Race:    {meta.get("race", "N/A")}\n\n'
        "The model's predicted cumulative risk of lung cancer development "
        'for this patient over the next 6 years is:\n'
        f'{pred_lines}\n\n'
        'Follow-up:\n'
        f'{cancer_text}\n\n'
        "The model's attention is focused on the highlighted regions (red/cyan colored) "
        'as shown below:'
    )

    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: text ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_xy(0, 0)
    pdf.set_font('Arial', 'B', 23.0)
    pdf.cell(w=210, h=22,
             txt=f'TANGERINE Report for CT Scan {patient_id}',
             ln=1, border=0, align='C', fill=False)

    pdf.set_font('Arial', 'B', 13.0)
    pdf.multi_cell(w=200, h=8, txt=body, border=0, align='L', fill=False)

    # ── Attention map: load and get top-3 slices ──────────────────────
    attn_path = attn_dir / patient_id / 'attention_rollout.nii.gz'

    # CT path: try images_dir/{patient_id}.nii.gz or {patient_id}_preprocessed.nii.gz
    ct_path = None
    if images_dir is not None:
        for suffix in ['.nii.gz', '_preprocessed.nii.gz']:
            candidate = Path(images_dir) / f'{patient_id}{suffix}'
            if candidate.exists():
                ct_path = candidate
                break

    grid_img, top3 = make_key_frame_grid(attn_path, ct_path)

    # ── Append top-3 attention slices to Page 1 ───────────────────────
    if top3:
        top3_lines = '\n'.join(
            f'  Rank {i+1}: axial slice {s_idx:>4d} / 256'
            f'   (mean attention: {score:.4f})'
            for i, (s_idx, score) in enumerate(top3)
        )
        pdf.set_font('Arial', 'B', 13.0)
        pdf.multi_cell(w=200, h=8,
                       txt='\nModel attention peak region (slices to review in PACS):\n'
                           + top3_lines,
                       border=0, align='L', fill=False)

    # ── Page 2: attention grid ─────────────────────────────────────────
    if grid_img is not None:
        tmp_img = str(out_pdf).replace('.pdf', '_tmp_grid.png')
        grid_img.save(tmp_img)
        pdf.add_page()
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(w=210, h=10,
                 txt='Attention rollout - 9 representative axial slices'
                     ' (lung window CT + attention overlay)',
                 ln=1, align='C')
        pdf.set_font('Arial', '', 9)
        pdf.cell(w=210, h=6,
                 txt='Red label = slice index.  Yellow label = mean attention intensity on slice.',
                 ln=1, align='C')
        pdf.image(tmp_img, x=0, y=22, w=210, h=210)
        os.remove(tmp_img)
    else:
        pdf.add_page()
        pdf.set_font('Arial', '', 12)
        pdf.cell(w=210, h=20,
                 txt='Attention map not available for this patient.',
                 ln=1, align='C')

    pdf.output(str(out_pdf), 'F')


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    run_dir    = Path(args.run_dir)
    attn_dir   = run_dir / 'attention' / 'rollout' / 'by_ct'
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    preds_csv = run_dir / 'test_predictions.csv'
    if not preds_csv.exists():
        raise FileNotFoundError(f'test_predictions.csv not found in {run_dir}')
    preds_df = pd.read_csv(preds_csv)
    preds_df['patient_id'] = preds_df['patient_id'].apply(
        lambda x: str(int(float(x))))
    print(f'Loaded {len(preds_df)} test predictions')

    # Load metadata (optional — provides sex/age/race/smoke/dates)
    meta_df = None
    if args.meta_csv and Path(args.meta_csv).exists():
        meta_df = pd.read_csv(args.meta_csv)
        # Anonymised accession number is the patient key
        meta_key = 'anon_acc' if 'anon_acc' in meta_df.columns else meta_df.columns[0]
        meta_df[meta_key] = meta_df[meta_key].apply(lambda x: str(int(float(x))))
        print(f'Loaded metadata: {len(meta_df)} rows  (key: {meta_key})')
    else:
        print('No metadata CSV provided — sex/age/race will show as N/A')

    # Filter to cancer patients (or all if --all_patients)
    if args.all_patients:
        target_df = preds_df
    else:
        target_df = preds_df[preds_df['cancer'] == 1].reset_index(drop=True)
    print(f'Generating reports for {len(target_df)} patients '
          f'({"all" if args.all_patients else "cancer=1 only"})')

    for _, row in target_df.iterrows():
        pid = str(row['patient_id'])

        # Merge metadata
        if meta_df is not None:
            meta_rows = meta_df[meta_df[meta_key] == pid]
            meta = meta_rows.iloc[0].to_dict() if len(meta_rows) > 0 else {}
        else:
            meta = {}
        # Fallback: use what's in predictions CSV
        meta.setdefault('cancer',       int(row.get('cancer', 0)))
        meta.setdefault('time_at_event', float(row.get('time_at_event', 0)))

        preds = {f'pred_{t}': float(row.get(f'pred_{t}', 0)) for t in range(1, 7)}

        out_pdf = out_dir / f'ct_{pid}_report.pdf'
        try:
            build_report(
                patient_id=pid,
                preds=preds,
                meta=meta,
                attn_dir=attn_dir,
                images_dir=args.images_dir,
                out_pdf=out_pdf,
            )
            print(f'  ✅  {pid}  →  {out_pdf.name}')
        except Exception as e:
            print(f'  ❌  {pid}  →  ERROR: {e}')

    print(f'\nDone. Reports saved to: {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir',      required=True,
                   help='Path to run output dir (contains test_predictions.csv + attention/)')
    p.add_argument('--meta_csv',     default=None,
                   help='Path to original metadata CSV (for sex/age/race/dates)')
    p.add_argument('--images_dir',   default=None,
                   help='Path to preprocessed CT NIfTI files (for CT+attention overlay)')
    p.add_argument('--output_dir',   default=None,
                   help='Where to save PDF reports (default: run_dir/reports)')
    p.add_argument('--all_patients', action='store_true', default=False,
                   help='Generate reports for all test patients, not just cancer=1')
    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.run_dir) / 'attention' / 'rollout' / 'reports')

    main(args)
