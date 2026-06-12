#!/usr/bin/env python3
"""
Generate training scripts for all 5 folds from template
"""
import re
from pathlib import Path

TEMPLATE_FILE = "train_fold0.sh"
FOLD_RANGE = range(5)

def main():
    # Read template
    template_path = Path(TEMPLATE_FILE)
    if not template_path.exists():
        print(f"✗ ERROR: Template {TEMPLATE_FILE} not found")
        return False

    with open(template_path, 'r') as f:
        template = f.read()

    print(f"Generating training scripts from {TEMPLATE_FILE}...")
    print("")

    created_files = []
    for fold_idx in FOLD_RANGE:
        # Replace fold index in template
        script_content = template
        script_content = re.sub(r'#SBATCH --job-name=tangerine_fold\d+',
                                f'#SBATCH --job-name=tangerine_fold{fold_idx}',
                                script_content)
        script_content = re.sub(r'FOLD=\d+', f'FOLD={fold_idx}', script_content)
        script_content = re.sub(r'logs/tangerine_fold\d+_',
                                f'logs/tangerine_fold{fold_idx}_',
                                script_content)
        script_content = re.sub(r'TANGERINE - 5-Fold CV: Fold \d+',
                                f'TANGERINE - 5-Fold CV: Fold {fold_idx}',
                                script_content)
        script_content = re.sub(r'Dataset splits not found - REGENERATING Fold \d+',
                                f'Dataset splits not found - REGENERATING Fold {fold_idx}',
                                script_content)
        script_content = re.sub(r'TRAINING FOLD \d+ WITH',
                                f'TRAINING FOLD {fold_idx} WITH',
                                script_content)
        script_content = re.sub(r'TRAINING FAILED FOR FOLD \d+',
                                f'TRAINING FAILED FOR FOLD {fold_idx}',
                                script_content)
        script_content = re.sub(r'TRAINING COMPLETE FOR FOLD \d+',
                                f'TRAINING COMPLETE FOR FOLD {fold_idx}',
                                script_content)
        script_content = re.sub(r'FOLD \d+ COMPLETED',
                                f'FOLD {fold_idx} COMPLETED',
                                script_content)

        # Write script
        output_file = f"train_fold{fold_idx}.sh"
        with open(output_file, 'w') as f:
            f.write(script_content)

        # Make executable
        Path(output_file).chmod(0o755)
        created_files.append(output_file)
        print(f"✓ Created {output_file}")

    print("")
    print("✅ Successfully generated all fold scripts!")
    print("")
    print("Next steps:")
    print("  1. Submit individual folds:")
    for fold_idx in FOLD_RANGE:
        print(f"       sbatch train_fold{fold_idx}.sh")
    print("")
    print("  2. Or submit all at once:")
    print("       for i in {{0..4}}; do sbatch train_fold$i.sh; done")
    print("")

    return True

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
