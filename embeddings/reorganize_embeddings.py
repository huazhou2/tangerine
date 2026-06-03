"""
Reorganize embeddings into structured folders:
  embeddings/
  ├── trained/        (fine-tuned model)
  ├── pretrain/       (pretrained model)
  └── combined/       (combined visualization plots)
"""

import os
import shutil
from pathlib import Path
import argparse

RACE_SHORTMAP = {
    'American Indian or Alaska Native': 'Am. Indian',
    'Native Hawaiian or Pacific Islander': 'Pac. Islander',
    'Not Reported': 'Not Reported',
    'Unknown': 'Unknown',
}

def shorten_race_in_csv(csv_file):
    """Replace long race labels with shortened versions."""
    import pandas as pd

    df = pd.read_csv(csv_file)
    if 'lrads_category' in df.columns:
        df['lrads_category'] = df['lrads_category'].replace(RACE_SHORTMAP)
        df.to_csv(csv_file, index=False)
        print(f"  ✓ Shortened race labels in {Path(csv_file).name}")


def reorganize_embeddings(embeddings_dir):
    """
    Reorganize embeddings into subfolder structure.

    Expected input structure:
      embeddings/
      ├── embeddings_layer*.npy
      ├── embeddings_meta_layer*.csv
      ├── umap_*.png
      └── umap_coords_layer*.npy

    Output structure:
      embeddings/
      ├── trained/
      │   ├── embeddings_layer*.npy
      │   ├── embeddings_meta_layer*.csv
      │   ├── umap_*_layer_final.png
      │   └── umap_coords_layer_final.npy
      ├── pretrain/
      │   ├── embeddings_layer0-23.npy
      │   ├── embeddings_meta_layer0-23.csv
      │   ├── umap_*_layer0-23.png
      │   └── umap_coords_layer0-23.npy
      └── combined/
          ├── umap_combined_layer*.png (pretrain only)
          └── umap_combined_layer_final.png (trained)
    """

    embeddings_path = Path(embeddings_dir)

    # Create subdirectories
    trained_dir = embeddings_path / 'trained'
    pretrain_dir = embeddings_path / 'pretrain'
    combined_dir = embeddings_path / 'combined'

    trained_dir.mkdir(exist_ok=True)
    pretrain_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)

    print(f"Created folders:")
    print(f"  ✓ {trained_dir.name}/")
    print(f"  ✓ {pretrain_dir.name}/")
    print(f"  ✓ {combined_dir.name}/")
    print()

    # Process files
    processed = {'trained': [], 'pretrain': [], 'combined': []}

    for file in embeddings_path.glob('*'):
        if file.is_dir():
            continue

        fname = file.name

        # Identify file type and destination
        if 'layer_final' in fname:
            # Fine-tuned model files
            dest = trained_dir / fname
            category = 'trained'
        elif fname.startswith('embeddings_layer') and fname.endswith('.npy'):
            # Check if it's a number (pretrain) or contains "final"
            if 'final' not in fname:
                dest = pretrain_dir / fname
                category = 'pretrain'
            else:
                dest = trained_dir / fname
                category = 'trained'
        elif fname.startswith('embeddings_meta_layer') and fname.endswith('.csv'):
            # Check layer number
            layer_num = fname.split('layer')[1].split('.')[0]
            if layer_num.isdigit() and int(layer_num) < 24:
                dest = pretrain_dir / fname
                category = 'pretrain'
            elif 'final' in fname:
                dest = trained_dir / fname
                category = 'trained'
            else:
                continue
        elif fname.startswith('umap_combined'):
            # Combined plots
            dest = combined_dir / fname
            category = 'combined'
        elif fname.startswith('umap_') and fname.endswith('.png'):
            # Individual variable UMAPs
            if 'layer_final' in fname:
                dest = trained_dir / fname
                category = 'trained'
            elif any(f'layer{i}.png' in fname for i in range(24)):
                dest = pretrain_dir / fname
                category = 'pretrain'
            else:
                continue
        elif fname.startswith('umap_coords'):
            # UMAP coordinates
            if 'final' in fname:
                dest = trained_dir / fname
                category = 'trained'
            else:
                dest = pretrain_dir / fname
                category = 'pretrain'
        else:
            continue

        # Copy file
        if file != dest:  # Only copy if different
            shutil.copy2(file, dest)
            processed[category].append(fname)

            # Shorten race labels in metadata CSVs
            if fname.endswith('.csv'):
                shorten_race_in_csv(dest)

    # Print summary
    print("Reorganization complete:")
    print(f"\n  📁 trained/: {len(processed['trained'])} files")
    print(f"  📁 pretrain/: {len(processed['pretrain'])} files")
    print(f"  📁 combined/: {len(processed['combined'])} files")

    # Optional: Remove original files from root
    print("\n⚠️  Original files remain in root embeddings/ directory.")
    print("   (kept for safety - delete manually if space needed)")


def main():
    parser = argparse.ArgumentParser(
        description='Reorganize embeddings into trained/pretrain/combined folders'
    )
    parser.add_argument('--embeddings_dir', type=str, required=True,
                       help='Path to embeddings directory')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove original files after reorganization')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("EMBEDDINGS REORGANIZATION")
    print(f"{'='*80}\n")
    print(f"Input directory: {args.embeddings_dir}\n")

    reorganize_embeddings(args.embeddings_dir)

    if args.cleanup:
        print("\n🧹 Cleaning up original files...")
        embeddings_path = Path(args.embeddings_dir)
        for file in embeddings_path.glob('embeddings_*.npy'):
            file.unlink()
        for file in embeddings_path.glob('embeddings_meta_*.csv'):
            file.unlink()
        for file in embeddings_path.glob('umap_*.png'):
            file.unlink()
        for file in embeddings_path.glob('umap_coords_*.npy'):
            file.unlink()
        print("✓ Original files deleted")

    print(f"\n{'='*80}")
    print("✅ REORGANIZATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
