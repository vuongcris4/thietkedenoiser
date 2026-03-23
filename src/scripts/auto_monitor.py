#!/usr/bin/env python3
"""
Auto-check tiến độ training và gửi notification.

Usage:
    python auto_monitor.py --interval 300  # Check mỗi 5 phút
"""
import argparse
import time
import sys
from datetime import datetime

try:
    import wandb
except ImportError:
    print("Error: wandb not installed.")
    sys.exit(1)

MODELS = ['lightweight', 'unet_resnet34', 'unet_effnet']


def get_latest_run(entity, project, model_name):
    """Get latest run for a model."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    model_runs = [
        r for r in runs
        if r.config.get('model', {}).get('name') == model_name
    ]
    model_runs.sort(key=lambda r: r.created_at, reverse=True)

    return model_runs[0] if model_runs else None


def check_all_models(entity, project):
    """Check status of all models."""
    results = []

    for model in MODELS:
        run = get_latest_run(entity, project, model)

        if not run:
            results.append({
                'model': model,
                'status': 'NOT_FOUND',
                'epoch': None,
                'val_miou': None,
            })
            continue

        # Get latest metrics from history
        try:
            history = run.history(
                keys=['epoch', 'val/mIoU', 'train/loss'],
                samples=1
            )

            if len(history) > 0:
                last_row = history.iloc[-1]
                results.append({
                    'model': model,
                    'status': run.state,
                    'epoch': int(last_row.get('epoch', 0)),
                    'val_miou': last_row.get('val/mIoU'),
                    'run_name': run.name,
                    'run_url': run.url,
                })
            else:
                results.append({
                    'model': model,
                    'status': run.state,
                    'epoch': None,
                    'val_miou': None,
                })
        except Exception as e:
            results.append({
                'model': model,
                'status': 'ERROR',
                'error': str(e),
            })

    return results


def print_status(results):
    """Print formatted status table."""
    print("\n" + "="*80)
    print(f"Experiment Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print(f"\n{'Model':<20} | {'Status':<12} | {'Epoch':>8} | {'Val mIoU':>10} | Run Name")
    print("-"*80)

    for r in results:
        model = r['model'][:18]
        status = r.get('status', 'UNKNOWN')[:10]
        epoch = str(r.get('epoch', 'N/A'))
        val_miou = f"{r.get('val_miou', 0):.4f}" if r.get('val_miou') else 'N/A'
        run_name = r.get('run_name', 'N/A')[:30]

        print(f"{model:<20} | {status:<12} | {epoch:>8} | {val_miou:>10} | {run_name}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Auto-monitor experiments')
    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--project', type=str, default='thietkedenoiser')
    parser.add_argument('--interval', type=int, default=300,
                        help='Check interval in seconds (default: 300)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')

    args = parser.parse_args()

    entity = args.entity or wandb.api.default_entity

    print(f"Starting auto-monitor for {entity}/{args.project}")
    print(f"Checking every {args.interval} seconds...")
    print(f"Models: {', '.join(MODELS)}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            results = check_all_models(entity, args.project)
            print_status(results)

            if args.once:
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")


if __name__ == '__main__':
    main()
