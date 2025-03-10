import os
import cv2
import subprocess
from tqdm import tqdm

#########################################################################
input_dir = "PATH/TO/INPUT/DIRECTORY/EGO4D/V1/FULL_SCALE"
output_dir = "PATH/TO/OUTPUT/DIRECTORY"
video_list_path = "./Hand100M/ego4d_name_list.txt"
#########################################################################

# Check if the video list file exists
if not os.path.isfile(video_list_path):
    print(f"Error: Video list file {video_list_path} does not exist!")
    exit(1)

# Check if the input video directory exists
if not os.path.isdir(input_dir):
    print(f"Error: Input video directory {input_dir} does not exist!")
    exit(1)

# Read the video list file
with open(video_list_path, "r") as file:
    lines = file.readlines()

# Skip the header line
video_lines = lines[1:]

# Count the number of videos in the list
video_count_in_list = len(video_lines)

# Count the number of videos in the input directory
video_count_in_dir = len([f for f in os.listdir(input_dir) if f.endswith(".mp4")])

# Check if the counts match
if video_count_in_list != video_count_in_dir:
    print(f"Error: Number of videos in the list ({video_count_in_list}) does not match the number of videos in the directory ({video_count_in_dir})!")
    exit(1)

print("Check passed: The number of videos in the list matches the number of videos in the directory. Starting processing...")

# Process each video
for line in tqdm(video_lines, desc="Processing videos", unit="video"):
    parts = line.strip().split()
    video_name = parts[1]  # example "0b0c7c26-8f38-4a22-8a9e-b99e4ae334fc.mp4"
    original_frame_count = int(parts[5])  # Original frame count
    setting_frame_count = int(parts[6])   # Setting frame count
    
    video_path = os.path.join(input_dir, video_name)
    video_id = video_name.replace(".mp4", "")
    video_output_dir = os.path.join(output_dir, video_id)
    os.makedirs(video_output_dir, exist_ok=True)

    # Extract frames from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open the video : {video_path}")
        continue

    frame_idx = 0
    saved_frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break if the frame is not read correctly
        
        if frame_idx % 30 == 0:  # Every 30 frames save one
            frame_filename = f"frame_{saved_frame_idx:06d}.jpg"
            frame_path = os.path.join(video_output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_idx += 30
        
        frame_idx += 1

    cap.release()
    print(f"Finish to process video : {video_name}, save frame num: {saved_frame_idx // 30}")
    
print("All videos processed successfully!")