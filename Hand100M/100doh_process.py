import os
import cv2
import subprocess
from tqdm import tqdm

#########################################################################
input_dir = "PATH/TO/INPUT/DIRECTORY/100DOH/VIDEO"
output_dir = "PATH/TO/OUTPUT/DIRECTORY"
video_list_path = "./Hand100M/100doh_valid_filter/100doh_valid_name_list.txt"   # only for valid videos
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

# Process each video
for line in tqdm(lines, desc="Processing videos", unit="video"):
    # Extract video information
    video_info = line.split()
    video_id = video_info[0]  # Video ID
    frame_rate = video_info[4]  # Frame rate (e.g., "30/1" or "30000/1001")
    resolution = video_info[3]  # Resolution (e.g., "1280x720")

    # Calculate the frame rate as a float
    if "/" in frame_rate:
        numerator, denominator = frame_rate.split("/")
        frame_rate_float = float(numerator) / float(denominator)
    else:
        frame_rate_float = float(frame_rate)

    # Construct the video file path
    video_path = os.path.join(input_dir, f"{video_id}.mp4")
    if not os.path.isfile(video_path):
        print(f"Warning: Video file {video_id}.mp4 does not exist. Skipping!")
        continue

    # Create the output directory
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
    print(f"Finish to process video : {video_id}.mp4, save frame num: {saved_frame_idx // 30}")

print("All videos processed successfully!")