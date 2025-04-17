#!/usr/bin/env python3
import os
import re
import cv2
import json
import argparse
import pandas as pd

# ------------------------------
# Configuration (adjust if needed)
# ------------------------------
VIDEO_FPS = 10  # Frames per second for the output video

# ------------------------------
# Function to convert a folder of PNG frames to an MP4 video.
# ------------------------------
def convert_images_to_video(episode_dir, output_path, fps=VIDEO_FPS):
    # List PNG files (ignoring files with .meta in the name)
    images = sorted(
        [f for f in os.listdir(episode_dir)
         if f.endswith('.png') and not f.endswith('.meta')]
    )
    
    if not images:
        print(f"No PNG images found in {episode_dir}. Skipping video conversion.")
        return

    first_image_path = os.path.join(episode_dir, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Could not read the first frame in {episode_dir}. Skipping.")
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(episode_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read frame {img_path}. Skipping.")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved video: {output_path}")

# ------------------------------
# Function to convert a JSONL file to a Parquet file.
# ------------------------------
def convert_jsonl_to_parquet(jsonl_file, parquet_file):
    records = []
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    records.append(record)
    except Exception as e:
        print(f"Error reading {jsonl_file}: {e}")
        return

    if records:
        df = pd.DataFrame(records)
        try:
            df.to_parquet(parquet_file, index=False)
            print(f"Saved parquet: {parquet_file}")
        except Exception as e:
            print(f"Error writing {parquet_file}: {e}")
    else:
        print(f"No records found in {jsonl_file}. Skipping conversion.")

# ------------------------------
# Main processing function.
# ------------------------------
def main():
    # Define default directories for data and videos.
    default_input_data_dir = r"C:\Users\trege\RL Sim\Assets\RESULTS\Data"
    default_input_videos_dir = r"C:\Users\trege\RL Sim\Assets\RESULTS\videos"
    default_output_dir = "New_results_0001"  # Relative output directory

    parser = argparse.ArgumentParser(
        description="Convert Unity-generated JSONL files and image sequences to Parquet and MP4 respectively."
    )
    parser.add_argument("--input-data-dir", type=str, default=default_input_data_dir,
                        help=fr"Path to input data folder (JSONL files). Default: {default_input_data_dir}")
    parser.add_argument("--input-videos-dir", type=str, default=default_input_videos_dir,
                        help=fr"Path to input videos folder (PNG sequences). Default: {default_input_videos_dir}")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help=f"Path to output directory where the new file structure will be created. Default: {default_output_dir}")
    
    args = parser.parse_args()

    # Create output directories if they do not exist.
    output_data_dir = os.path.join(args.output_dir, "data")
    output_videos_dir = os.path.join(args.output_dir, "videos")
    output_meta_dir = os.path.join(args.output_dir, "meta")
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(output_videos_dir, exist_ok=True)
    os.makedirs(output_meta_dir, exist_ok=True)

    # ------------------------------
    # Process video files: Convert PNG sequences to MP4.
    # ------------------------------
    for root, dirs, files in os.walk(args.input_videos_dir):
        if any(f.endswith('.png') and not f.endswith('.meta') for f in files):
            # Assume that the current directory (root) is the episode folder.
            # We want to place the resulting video file into the parent folder.
            episode_name = os.path.basename(root)
            parent_dir = os.path.dirname(root)
            
            # Compute relative path from the input videos folder for the parent folder.
            rel_parent = os.path.relpath(parent_dir, args.input_videos_dir)
            out_video_folder = os.path.join(output_videos_dir, rel_parent)
            os.makedirs(out_video_folder, exist_ok=True)

            # The video file name is based on the episode folder name.
            output_video = os.path.join(out_video_folder, f"{episode_name}.mp4")
            if os.path.exists(output_video):
                print(f"Video already exists: {output_video}. Skipping.")
            else:
                convert_images_to_video(root, output_video)

    # ------------------------------
    # Process data files: Convert JSONL files to Parquet and collect episode info.
    # ------------------------------
    episodes_info = []
    for root, dirs, files in os.walk(args.input_data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, args.input_data_dir)
                out_data_subdir = os.path.join(output_data_dir, rel_path)
                os.makedirs(out_data_subdir, exist_ok=True)
                parquet_path = os.path.join(out_data_subdir, file.replace('.jsonl', '.parquet'))
                if os.path.exists(parquet_path):
                    print(f"Parquet file already exists: {parquet_path}. Skipping.")
                else:
                    # Initialize set for unique task indices.
                    task_indices = set()
                    record_count = 0

                    # Read each line, counting non-empty lines and gathering task indices.
                    with open(jsonl_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                record_count += 1
                                try:
                                    record = json.loads(line)
                                    if "task_index" in record:
                                        task_indices.add(record["task_index"])
                                except Exception as e:
                                    print(f"Error parsing a line in {jsonl_path}: {e}")

                    convert_jsonl_to_parquet(jsonl_path, parquet_path)
                    
                    # Extract episode index from the filename (assuming 'episode_XXXX' format).
                    match = re.search(r'episode_(\d+)', file)
                    episode_index = int(match.group(1)) if match else None
                    
                    episodes_info.append({
                        "episode_index": episode_index,
                        "tasks": sorted(list(task_indices)),
                        "length": record_count
                    })

    # Save the episode summary as a JSONL file in the meta folder.
    episodes_summary_path = os.path.join(output_meta_dir, "episodes.jsonl")
    try:
        with open(episodes_summary_path, 'w') as f:
            for episode in episodes_info:
                f.write(json.dumps(episode) + "\n")
        print(f"Saved episode summary to {episodes_summary_path}")
    except Exception as e:
        print(f"Error writing episode summary: {e}")

    # ------------------------------
    # Update info.json with new parameters.
    # ------------------------------
    info_path = os.path.join(output_meta_dir, "info.json")
    # Load existing info.json if it exists; otherwise, create a default structure.
    if os.path.exists(info_path):
        try:
            with open(info_path, 'r') as f:
                info = json.load(f)
        except Exception as e:
            print(f"Error reading info.json: {e}")
            info = {}
    else:
        info = {
            "codebase_version": "v2.0",
            "robot_type": "two_wheel",
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": 0,
            "fps": 10.0,
            "splits": {
                "train": "0:40",
                "val": "40:50"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {}  # You can include feature details here if needed.
        }

    # Update statistics.
    num_episodes = len(episodes_info)
    total_frames = sum(ep["length"] for ep in episodes_info)
    
    # Count the total number of MP4 video files in the output videos directory.
    total_videos = 0
    for root, dirs, files in os.walk(output_videos_dir):
        for f in files:
            if f.endswith('.mp4'):
                total_videos += 1

    info["total_episodes"] = num_episodes
    info["total_frames"] = total_frames
    info["total_videos"] = total_videos
    info["total_chunks"] = 1               # Assuming all episodes fall into one chunk.
    info["chunks_size"] = num_episodes       # Chunk size equals the number of episodes.

    try:
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Updated info.json: {info_path}")
    except Exception as e:
        print(f"Error writing info.json: {e}")

if __name__ == "__main__":
    main()
