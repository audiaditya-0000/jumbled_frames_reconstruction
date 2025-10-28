from skimage.metrics import structural_similarity as ssim
import cv2
import os
import numpy as np
from multiprocessing import Pool, cpu_count

def compute_row(args):
    i, frames = args
    n = len(frames)
    row = np.zeros(n)
    for j in range(i, n):
        row[j] = ssim(frames[i], frames[j])
    return i, row

def get_frames(video,frame_folder):
     #this function used to get the frames from the video
    vid = cv2.VideoCapture(video)
    count, success = 0, True
    while success:
        success, image = vid.read()
        if success:
            filename = os.path.join(frame_folder, f"frame{count:03d}.jpg")
            result = cv2.imwrite(filename, image)
            if result:
                print(f"Saved frame {count}")
            else:
                print(f"Failed to save frame {count}")
            count += 1

    vid.release()
    print("frames extracted")

def similarity(frame_folder):
    #this function used to compute the similarity matrix between frames
    frame_files = sorted(os.listdir(frame_folder))
    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(frame_folder, f))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (320, 180))  
        frames.append(img_gray)

    n = len(frames)
    sim_matrix = np.zeros((n, n))

    args = [(i, frames) for i in range(n)]

    with Pool(cpu_count()) as pool:
        results = pool.map(compute_row, args)

    for i, row in results:
        sim_matrix[i, i:] = row[i:]
        sim_matrix[i:, i] = row[i:]

    return sim_matrix

def correct_order(sim_matrix):
    #returns the correct order of frames based on the similarity matrix
    n = sim_matrix.shape[0]
    used = set()
    order = []

    current = 0
    order.append(current)
    used.add(current)

    for _ in range(n - 1):
        sims = sim_matrix[current]
        next_frame = None
        max_sim = -1
        for i in range(n):
            if i not in used and sims[i] > max_sim:
                max_sim = sims[i]
                next_frame = i
        order.append(next_frame)
        used.add(next_frame)
        current = next_frame

    return order


def reconstruct(ordered_frames,frame_folder,output_path,fps=30):
    #this function reconstructs the final correct video
    first_frame_path = os.path.join(frame_folder, f"frame{ordered_frames[0]:03d}.jpg")
    frame = cv2.imread(first_frame_path)
    height, width, layers = frame.shape
    size = (width, height)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for idx in ordered_frames:
        frame_path = os.path.join(frame_folder, f"frame{idx:03d}.jpg")
        img = cv2.imread(frame_path)
        out.write(img)

    out.release()
    print(f"Video reconstructed and saved to '{output_path}'")

def main():
    video="data/jumbled_video.mp4"
    frame_folder="data/frame_folder"
    output="data/reconstructed_video.mp4"

    get_frames(video,frame_folder)
    similarity_matrix=similarity(frame_folder)
    ordered_frames=correct_order(similarity_matrix)
    reconstruct(ordered_frames,frame_folder,output)

if __name__ == "__main__":
    main()