from video_sampler import CreateSample

# Usage
input_dir = '/home/s2186747/data/project/toybox'
output_dir = '/home/s2186747/data/project/toybox_sample'
frame_interval = 3
test_objects_per_category = 3

# Initialize the CreateSample object and process videos
sampler = CreateSample(input_dir, output_dir, frame_interval, test_objects_per_category)
sampler.split_objects()
sampler.process_videos()
