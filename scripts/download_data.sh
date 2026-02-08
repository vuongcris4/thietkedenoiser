#!/bin/bash
set -e
echo "=== Downloading OpenEarthMap ==="
cd ~/thietkedenoiser/data

# Option 1: From Zenodo (official)
if [ ! -d "OpenEarthMap" ]; then
    wget -c https://zenodo.org/records/7223446/files/OpenEarthMap.zip
    unzip OpenEarthMap.zip
    echo "Dataset downloaded and extracted"
else
    echo "Dataset already exists"
fi

echo "=== Dataset stats ==="
find OpenEarthMap -name '*.tif' | wc -l
ls OpenEarthMap/
