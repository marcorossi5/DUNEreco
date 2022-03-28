#!/bin/bash
# Usage: ./run_plot_example.sh

mkdir -p plot_example
tar -xf dunetpc_inspired_v09_p2GeV_rawdigits.tar.gz -C plot_example
python plot_event_example.py