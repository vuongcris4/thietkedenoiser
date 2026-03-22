#!/bin/bash
# =============================================================================
# Sub-agent: Check tiến độ training experiments
# =============================================================================
# Usage: ./check_agent.sh [model_name]
#
# Examples:
#   ./check_agent.sh                    # Check tất cả models
#   ./check_agent.sh lightweight        # Check lightweight model only
# =============================================================================

cd "$(dirname "$0")"

echo "============================================================"
echo "Checking experiment progress..."
echo "Time: $(date)"
echo "============================================================"

# Run progress check
python check_progress.py $@

echo ""
echo "============================================================"
echo "Check complete!"
echo "============================================================"
