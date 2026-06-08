#!/usr/bin/env bash
# Run full backtest: all 3 agents, print metrics table
set -e
cd "$(dirname "$0")/.."
python backtest/src/backtest_engine.py --config configs/default.yaml --results results
