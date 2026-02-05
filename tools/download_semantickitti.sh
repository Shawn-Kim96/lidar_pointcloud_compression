#!/bin/bash

# SemanticKITTI Download Helper
# Note: SemanticKITTI is hosted on the KITTI benchmark website and requires registration.
# It cannot be downloaded via a simple public URL without a specialized cookie or auth.

DATA_DIR="../data/semantickitti"
mkdir -p $DATA_DIR

echo "=========================================================="
echo "      SemanticKITTI Dataset Download Helper"
echo "=========================================================="
echo ""
echo "1. Go to: http://www.semantic-kitti.org/dataset.html#download"
echo "2. Click 'Download Odometry Validation set' (and Training set)"
echo "   (This will redirect you to the KITTI Vision Benchmark Suite)"
echo "   You need to register/login to download 'data_odometry_velodyne.zip' (80GB)"
echo "   and 'data_odometry_labels.zip' (179MB)."
echo ""
echo "3. Once downloaded, place the .zip files in: "
echo "   $(pwd)/$DATA_DIR"
echo ""
echo "=========================================================="
echo "Checking for existing zip files..."

if [ -f "$DATA_DIR/data_odometry_velodyne.zip" ] && [ -f "$DATA_DIR/data_odometry_labels.zip" ]; then
    echo "Found zip files! Extracting..."
    unzip -d $DATA_DIR $DATA_DIR/data_odometry_velodyne.zip
    unzip -d $DATA_DIR $DATA_DIR/data_odometry_labels.zip
    echo "Extraction complete. Dataset is ready at $DATA_DIR/dataset"
else
    echo "Zip files not found in $DATA_DIR."
    echo "Please manually download them and run this script again to extract."
    echo "Or run: python ../train.py --data_dir $DATA_DIR/dataset/sequences"
fi

echo ""
echo "Recommended Structure:"
echo "$DATA_DIR/dataset/"
echo "  ├── sequences/"
echo "  │   ├── 00/"
echo "  │   │   ├── velodyne/  (.bin files)"
echo "  │   │   ├── labels/    (.label files)"
echo "  │   │   ├── calib.txt"
echo "  │   │   └── poses.txt"
echo "  │   ├── 01/"
echo "  │   └── ..."
