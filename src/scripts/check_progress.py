#!/usr/bin/env python3
"""
Check tiến độ training experiments từ W&B.

Usage:
    python check_progress.py              # Check tất cả experiments
    python check_progress.py --model lightweight  # Check 1 model cụ thể
"""
import argparse
import sys
import os

try:
    import wandb
    from wandb.sdk.wandb_run import Run
except ImportError:
    print("Error: wandb not installed. Install with: pip install wandb")
    sys.exit(1)

# 8 lớp phân loại trong OpenEarthMap
CLASS_NAMES = [
    'Bareland', 'Rangeland', 'Developed', 'Road',
    'Tree', 'Water', 'Agriculture', 'Building'
]

# Models to track
MODELS = ['lightweight', 'unet_resnet34', 'unet_effnet']


def get_model_runs(entity, project, model_name):
    """Get all runs for a specific model."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    model_runs = []
    for run in runs:
        if run.config.get('model', {}).get('name') == model_name:
            model_runs.append(run)

    # Sort by created_at, newest first
    model_runs.sort(key=lambda r: r.created_at, reverse=True)
    return model_runs


def print_run_summary(run, max_history=5):
    """Print summary for a single run."""
    print(f"\n  {'='*70}")
    print(f"  Run: {run.name}")
    print(f"  URL: {run.url}")
    print(f"  Status: {run.state}")
    print(f"  Created: {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    # Get config
    config = run.config
    epochs = config.get('training', {}).get('epochs', 'N/A')
    batch_size = config.get('training', {}).get('batch_size', 'N/A')
    lr = config.get('training', {}).get('lr', 'N/A')
    print(f"  Config: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # Get summary metrics
    summary = run.summary
    best_miou = summary.get('best_miou')
    if best_miou is None:
        # Try to get from history if not in summary
        try:
            history = run.history(keys=['val/mIoU'], samples=max_history)
            if len(history) > 0:
                best_miou = history['val/mIoU'].max()
        except:
            best_miou = 'N/A'

    print(f"  Best mIoU: {best_miou}")

    # Get last 5 epochs from history
    try:
        history = run.history(
            keys=['epoch', 'train/loss', 'val/loss', 'val/mIoU', 'lr'],
            samples=max_history
        )

        if len(history) > 0:
            print(f"\n  Last {min(len(history), max_history)} epochs:")
            print(f"  {'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'Val mIoU':>10} | {'LR':>12}")
            print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

            for _, row in history.tail(max_history).iterrows():
                epoch = int(row.get('epoch', 0))
                train_loss = row.get('train/loss', 0)
                val_loss = row.get('val/loss', 0)
                val_miou = row.get('val/mIoU', 0)
                lr = row.get('lr', 0)
                print(f"  {epoch:>6} | {train_loss:>12.4f} | {val_loss:>10.4f} | {val_miou:>10.4f} | {lr:>12.2e}")
    except Exception as e:
        print(f"  Could not retrieve history: {e}")


def check_progress(entity, project, model_name=None):
    """Check progress of experiments."""
    models_to_check = [model_name] if model_name else MODELS

    print("="*70)
    print(f"W&B Experiment Progress Check")
    print(f"Project: {entity}/{project}")
    print(f"Models: {', '.join(models_to_check)}")
    print("="*70)

    for model in models_to_check:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#'*70}")

        runs = get_model_runs(entity, project, model)

        if not runs:
            print(f"  No runs found for model '{model}'")
            continue

        print(f"  Found {len(runs)} run(s)")

        # Show latest 3 runs
        for run in runs[:3]:
            print_run_summary(run)


def main():
    parser = argparse.ArgumentParser(description='Check W&B experiment progress')
    parser.add_argument('--entity', type=str, default=None,
                        help='W&B entity (team/username)')
    parser.add_argument('--project', type=str, default='thietkedenoiser',
                        help='W&B project name')
    parser.add_argument('--model', type=str, choices=MODELS, default=None,
                        help='Check specific model only')

    args = parser.parse_args()

    # Get entity from wandb login if not provided
    entity = args.entity
    if not entity:
        try:
            entity = wandb.api.default_entity
        except:
            print("Error: Could not determine W&B entity. Please login with: wandb login")
            print("Or specify --entity")
            sys.exit(1)

    check_progress(entity, args.project, args.model)


if __name__ == '__main__':
    main()
