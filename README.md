# Jumbled_frames_reconstruction
Provided with a 10 second 30 fps video whose frames have been randomly jumbled, my goal was to reconstruct the original video as accurately and efficiently as possible.

## Algorithm Explanation
My project uses an image similarity based ordering algorithm that measures how visually similar each frames are to each other. I used SSIM because it compares images based on how humans actually perceive visual similarity, not just pixel differences. It helps identify which frames look most alike, which is perfect for figuring out the correct video order. It’s also simple to implement, doesn’t need any training data, and runs faster with multiprocessing

There are 5 main steps I followed
1. Frame Extraction 
The input video is decomposed into individual frames using OpenCV. Each frame is saved as a .jpg image for later comparison.

2. Feature Representation
Each frame is converted to grayscale to reduce computational cost and focus on structural details rather than color information.
The frames are resized to a uniform size (320×180) to standardize comparisons.

3. Similarity Computation
The Structural Similarity Index (SSIM) from scikit-image is used to compute pairwise similarity between frames. Each frame is compared with all others, forming a symmetric similarity matrix S, where S[i][j] = SSIM(frame_i, frame_j)

4. Ordering Algorithm
The algorithm begins by identifying the frame with the lowest total similarity, usually the first or last frame. Starting from this frame, the algorithm iteratively selects the next most similar unused frame, building a likely chronological order.
This greedy approach works well when neighboring frames are visually consistent, such as in natural videos with smooth transitions.

5. Video Reconstruction
Ordered frames are stitched back together using OpenCV’s VideoWriter, producing a reconstructed .mp4 file. Optionally, the video is also reversed to handle cases where the correct sequence might be inverted.

### Design Considerations
1. Accuracy vs. Efficiency
SSIM provides high accuracy in capturing visual coherence but is computationally heavier than basic metrics like MSE. To balance this, frames are resized and converted to grayscale before similarity computation, maintaining structure while reducing processing time.

2. Time Complexity
The similarity computation is O(n²) in nature, where n is the number of frames. Parallelization across multiple CPU cores significantly reduces this runtime bottleneck. 

3. Parallelism
Each process independently computes one row of the similarity matrix using Pool.map(). The results are aggregated into a global matrix once all processes complete, ensuring efficiency without data conflicts.

4. Greedy Frame Ordering
While not globally optimal, the greedy nearest-neighbor strategy is computationally efficient (O(n²) total) and yields strong results for smoothly transitioning videos. It avoids the overhead of complex graph or ML-based sequence reconstruction

5. Robustness
Since the direction of ordering might be reversed (depending on the starting frame), the system generates both reconstructed and reversed versions, allowing the user to verify which one is correct visually.

## Instructions
### Install the dependencies using
pip install numpy opencv-python scikit-image

### To run
Create the folder to hold frames under data folder, name it  
frame_folder

Place the jumbled video at  
data/jumbled_video.mp4

Move into project folder  
cd jumbled_frames_reconstruction

Run the main script  
python src/main.py

## Execution Time Log
Total Execution Time: 222.57 seconds

System specs
- CPU: Intel Core i5 (4 cores)
- RAM: 16 GB
- OS: Windows 10 (64 bit)
- Python: 3.11

## Expected Output
Once you run the main script, you will see two output videos (because result may be reversed sometimes)

### Output
frames extracted  
Video reconstructed and saved to 'data/reconstructed_video.mp4'  
video reversed successfully and saved to 'data/reconstructed_reversed.mp4'  
