#!/bin/bash
cd "$(dirname "$0")"
echo "Installing required packages..."
pip3 install opencv-python PyQt6 ultralytics numpy
echo "Done! Double-click run.command to start the app"
