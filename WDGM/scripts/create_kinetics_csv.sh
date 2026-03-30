#!/bin/bash

VIDEO_DIRS=(
)

CSV_FILE="./data/pretrain/train.csv"

VIDEO_EXTENSIONS=("mp4" "avi" "mkv" "mov" "wmv" "flv" "webm")

echo "Generating CSV file..."
> $CSV_FILE

total_count=0
for video_dir in "${VIDEO_DIRS[@]}"; do
    if [ -d "$video_dir" ]; then
        echo "Processing directory: $video_dir"
        dir_count=0
        
        for ext in "${VIDEO_EXTENSIONS[@]}"; do
            for video in "$video_dir"/*.$ext; do
                if [ -f "$video" ]; then
                    abs_path=$(realpath "$video")
                    echo "$abs_path,-1" >> $CSV_FILE
                    ((dir_count++))
                fi
            done
        done
        
        echo "  Found $dir_count videos in $video_dir"
        total_count=$((total_count + dir_count))
    else
        echo "Warning: Directory $video_dir does not exist"
    fi
done

echo "========================================"
echo "Total: Generated $total_count entries in $CSV_FILE"
echo "First few entries:"
head -5 $CSV_FILE