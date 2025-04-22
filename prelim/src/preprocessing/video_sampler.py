import cv2
import os
import random
from collections import defaultdict

class CreateSample:
    def __init__(self, input_dir, output_dir, frame_interval=3, test_objects_per_category=3):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.test_objects_per_category = test_objects_per_category
        self.train_videos = []
        self.test_videos = []

    def split_objects(self):
        # Traverse the directory for individual objects and group videos
        for supercategory in os.listdir(self.input_dir):
            supercategory_path = os.path.join(self.input_dir, supercategory)
            if not os.path.isdir(supercategory_path):
                continue

            toy_groups = defaultdict(list)
            for toy in os.listdir(supercategory_path):
                toy_path = os.path.join(supercategory_path, toy)
                if not os.path.isdir(toy_path):
                    continue
                base_name = toy.split('_')[0]
                toy_groups[base_name].append(toy_path)

            for base_name, toys in toy_groups.items():
                if len(toys) > self.test_objects_per_category:
                    test_toys = random.sample(toys, self.test_objects_per_category)
                    train_toys = [toy for toy in toys if toy not in test_toys]
                else:
                    test_toys = toys
                    train_toys = []

                for toy_path in train_toys:
                    videos = [
                        os.path.join(toy_path, video)
                        for video in os.listdir(toy_path)
                        if os.path.isfile(os.path.join(toy_path, video))
                        and video.endswith(".mp4")
                        and not video.endswith("absent.mp4")
                    ]
                    self.train_videos.extend(videos)

                for toy_path in test_toys:
                    videos = [
                        os.path.join(toy_path, video)
                        for video in os.listdir(toy_path)
                        if os.path.isfile(os.path.join(toy_path, video))
                        and video.endswith(".mp4")
                        and not video.endswith("absent.mp4")
                    ]
                    self.test_videos.extend(videos)
                
        print(f"Split completed: {len(self.train_videos)} training videos, {len(self.test_videos)} test videos")

    def extract_frames_from_video(self, video_path, is_test=False):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if the video opened successfully
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video frame rate and calculate the interval frame count
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_count = int(fps * self.frame_interval)
        
        # Create directory for video frames, starting with train/test at the top level
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        subset_dir = 'test' if is_test else 'train'
        
        # Maintain the full directory structure in the output path after the train/test level
        relative_path = os.path.relpath(os.path.dirname(video_path), self.input_dir)
        video_output_dir = os.path.join(self.output_dir, subset_dir, relative_path, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        
        frame_count = 0
        save_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Break the loop if there are no more frames
            if not ret:
                break
            
            # Save the frame every `frame_interval_count` frames
            if frame_count % frame_interval_count == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{save_count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)
                save_count += 1
            
            frame_count += 1
        
        # Release the video capture object
        cap.release()
        print(f"Extracted {save_count} frames from {video_name} ({'test' if is_test else 'train'})")

    def process_videos(self):
        # Process videos in training and test sets separately
        for video in self.train_videos:
            self.extract_frames_from_video(video, is_test=False)
        
        for video in self.test_videos:
            self.extract_frames_from_video(video, is_test=True)
