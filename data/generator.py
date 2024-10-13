"""
This module contains functions to generate various types of visual illusion datasets.
Each function generates images and corresponding labels for a specific type of illusion.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from PIL import Image
import pandas as pd
import pyllusion
from tqdm import tqdm
import imageio

def interpolate_states(start_value, end_value, interp_factor):
    """
    Linearly interpolate between start_value and end_value.
    """
    return start_value + interp_factor * (end_value - start_value)



def dataset01_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Hering Wundt illusion with smooth transitions.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=['name', 'label', 'start_slope', 'end_slope', 'start_step_size', 'end_step_size', 'bend'])
    
    for i in tqdm(range(size)):
        label = int(np.random.rand() < positive_ratio)
        start_slope = np.random.rand() * 5 + 1
        end_slope = np.random.rand() * 5 + 1
        start_step_size = np.random.rand() * 0.15 + 0.1
        end_step_size = np.random.rand() * 0.15 + 0.1
        bend = 0

        video_path = os.path.join(path, f'hering_video_{i}.mp4')
        fig, ax = plt.subplots(figsize=(4, 4), dpi=64)
        plt.axis('off')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            slope = interpolate_states(start_slope, end_slope, interp_factor)
            step_size = interpolate_states(start_step_size, end_step_size, interp_factor)

            ax.clear()
            for angle in np.arange(-slope, slope, step_size):
                ax.plot(np.arange(-2, 2, 0.01), angle * np.arange(-2, 2, 0.01), 'k')

            if label:
                ax.plot([-1, -1], [-4, 4], 'r', linewidth=2)
                ax.plot([1, 1], [-4, 4], 'r', linewidth=2)
            else:
                bend = 0
                while bend == 0:
                    bend = np.random.rand() * 0.08
                sign = 1 if np.random.rand() > 0.5 else -1
                ax.plot([-1, -(1-sign * bend), -1], [-4, 0, 4], 'r', linewidth=2)
                ax.plot([1, (1-sign * bend), 1], [-4, 0, 4], 'r', linewidth=2)

            # Convert current frame to image and add to video
            fig.canvas.draw()
            frame_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            frame_image = frame_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            video_writer.append_data(frame_image)

        video_writer.close()

        # Save metadata for the video
        label_df.loc[len(label_df)] = [f'hering_video_{i}.mp4', label, start_slope, end_slope, start_step_size, end_step_size, bend]
    
    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)


def dataset02_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Muller-Lyer illusion with smooth transitions in rotation.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=['name', 'label', 'value', 'start_rotation', 'end_rotation', 'Top_x1', 'Top_y1', 'Top_x2', 'Top_y2', 'Bottom_x1', 'Bottom_y1', 'Bottom_x2', 'Bottom_y2'])
    
    for i in tqdm(range(size)):
        label = int(np.random.rand() < positive_ratio)
        diff = 0 if label else np.random.randint(-500, 500) / 1000.0  # Ensure float division
        strength = -np.random.randint(25, 35)
        mullerlyer = pyllusion.MullerLyer(illusion_strength=strength, difference=diff, distance=np.random.randint(80, 120) / 100.0)  # Ensure float division
        start_rotation = float(np.random.randint(0, 180))  # Ensure float
        end_rotation = float(np.random.randint(0, 180))  # Ensure float

        # Create video path
        video_path = os.path.join(path, f'mullerlyer_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)  # Ensure interp_factor is a float
            current_rotation = interpolate_states(start_rotation, end_rotation, interp_factor)

            # Generate the image with the current rotation
            img = mullerlyer.to_image(width=128, height=128, outline=4).rotate(angle=current_rotation, fillcolor=(255, 255, 255, 255))
            fn = lambda x: 255 if x > 210 else 0
            img = img.convert("L").point(fn, mode='1')

            # Convert image to numpy array and add to video
            img_np = np.array(img)
            video_writer.append_data(img_np)

        video_writer.close()

        # Save metadata for the video
        dict = mullerlyer.get_parameters()
        label_df.loc[len(label_df)] = [f'mullerlyer_video_{i}.mp4', label, diff, start_rotation, end_rotation, dict['Top_x1'], dict['Top_y1'], dict['Top_x2'], dict['Top_y2'], dict['Bottom_x1'], dict['Bottom_y1'], dict['Bottom_x2'], dict['Bottom_y2']]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)


def dataset03_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Poggendorff illusion with smooth transitions in illusion strength.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=['name', 'label', 'start_strength', 'end_strength', 'diff', 'Left_x1', 'Left_y1', 'Left_x2', 'Left_y2', 'Right_x1', 'Right_y1', 'Right_x2', 'Right_y2', 'Angle', 'Rectangle_Height', 'Rectangle_Width'])
    
    for i in tqdm(range(size)):
        label = int(np.random.rand() < positive_ratio)
        diff = 0 if label else 0.3 * np.random.rand()
        start_strength = -np.random.randint(1, 60)
        end_strength = -np.random.randint(1, 60)
        
        # Create video path
        video_path = os.path.join(path, f'poggendorff_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            current_strength = interpolate_states(start_strength, end_strength, interp_factor)

            # Generate the image with the current illusion strength
            poggendorff = pyllusion.Poggendorff(illusion_strength=current_strength, difference=diff)
            img = poggendorff.to_image(width=128, height=128)
            fn = lambda x: 255 if x > 210 else 0
            img = img.convert("L").point(fn, mode='1')

            # Convert image to numpy array and add to video
            img_np = np.array(img)
            video_writer.append_data(img_np)

        video_writer.close()

        # Save metadata for the video
        dict = poggendorff.get_parameters()
        label_df.loc[len(label_df)] = [f'poggendorff_video_{i}.mp4', label, start_strength, end_strength, dict['Difference'], dict['Left_x1'], dict['Left_y1'], dict['Left_x2'], dict['Left_y2'], dict['Right_x1'], dict['Right_y1'], dict['Right_x2'], dict['Right_y2'], dict['Angle'], dict['Rectangle_Height'], dict['Rectangle_Width']]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)

def dataset04_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Vertical-Horizontal illusion with smooth transitions in illusion strength.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=['name', 'label', 'start_strength', 'end_strength', 'value'])

    for i in tqdm(range(size)):
        label = int(np.random.rand() < positive_ratio)
        diff = 0 if label else 0.3 * np.random.rand()
        start_strength = -np.random.randint(60, 90)
        end_strength = -np.random.randint(60, 90)
        
        # Create video path
        video_path = os.path.join(path, f'vertical_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            current_strength = interpolate_states(start_strength, end_strength, interp_factor)

            # Generate the image with the current illusion strength
            zollner = pyllusion.VerticalHorizontal(illusion_strength=current_strength, difference=diff)
            img = zollner.to_image(width=128, height=128)
            fn = lambda x: 255 if x > 210 else 0
            img = img.convert("L").point(fn, mode='1')

            # Convert image to numpy array and add to video
            img_np = np.array(img)
            video_writer.append_data(img_np)

        video_writer.close()

        # Save metadata for the video
        label_df.loc[len(label_df)] = [f'vertical_video_{i}.mp4', label, start_strength, end_strength, diff]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)


def dataset05_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Zollner illusion with smooth transitions in illusion strength.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=['name', 'label', 'start_strength', 'end_strength', 'value', 'rotation'])

    for i in tqdm(range(size)):
        label = int(np.random.rand() < positive_ratio)
        diff = 0 if label else 9 * np.random.rand()
        start_strength = np.random.randint(45, 65)
        end_strength = np.random.randint(45, 65)
        rotation = np.random.randint(0, 180)
        
        # Create video path
        video_path = os.path.join(path, f'zollner_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            current_strength = interpolate_states(start_strength, end_strength, interp_factor)

            # Generate the image with the current illusion strength
            zollner = pyllusion.Zollner(illusion_strength=current_strength, difference=diff)
            img = zollner.to_image(width=128, height=128).rotate(angle=rotation, fillcolor=(255, 255, 255, 255))
            fn = lambda x: 255 if x > 210 else 0
            img = img.convert("L").point(fn, mode='1')

            # Convert image to numpy array and add to video
            img_np = np.array(img)
            video_writer.append_data(img_np)

        video_writer.close()

        # Save metadata for the video
        label_df.loc[len(label_df)] = [f'zollner_video_{i}.mp4', label, start_strength, end_strength, diff, rotation]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)


def dataset06_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Red-Yellow Boundary illusion with smooth transitions in width, x, and y.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    label_df = pd.DataFrame(columns=["name", "label", "start_width", "end_width", "start_x", "end_x", "start_y", "end_y", "r", "g", "b"])

    for i in tqdm(range(size)):
        # Initial values
        start_x = np.random.rand() * 43 - 21
        start_y = np.random.rand() * 43 - 21
        end_x = np.random.rand() * 43 - 21
        end_y = np.random.rand() * 43 - 21
        start_width = np.min([np.abs(start_x + 32), np.abs(32 - start_x), np.abs(start_y + 32), np.abs(32 - start_y), np.max([np.random.rand() * 64, 11]), 42])
        end_width = np.min([np.abs(end_x + 32), np.abs(32 - end_x), np.abs(end_y + 32), np.abs(32 - end_y), np.max([np.random.rand() * 64, 11]), 42])

        # Determine color and label
        if np.random.rand() > positive_ratio:
            c = (1, np.random.rand(), 0)
            label = 0
        else:
            c = (1, 0.5, 0)
            label = 1

        # Create video path
        video_path = os.path.join(path, f'rect_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            current_x = interpolate_states(start_x, end_x, interp_factor)
            current_y = interpolate_states(start_y, end_y, interp_factor)
            current_width = interpolate_states(start_width, end_width, interp_factor)

            # Create a figure and draw the rectangle with interpolated parameters
            fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=16)
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.add_patch(
                patches.Rectangle(
                    (current_x, current_y),
                    width=current_width,
                    height=current_width,
                    color=c
                )
            )
            ax1.set_xlim([-32, 32])
            ax1.set_ylim([-32, 32])
            ax1.set_axis_off()

            # Convert figure to image and add to video
            fig.canvas.draw()
            img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            video_writer.append_data(img_np)

            plt.close(fig)  # Close the figure after each frame to free up memory

        video_writer.close()

        # Save metadata for the video
        label_df.loc[len(label_df)] = [f'rect_video_{i}.mp4', label, start_width, end_width, start_x, end_x, start_y, end_y, *c]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)


def dataset07_video(path, size, positive_ratio, video_frames=10, framerate=10):
    """
    Generates a video dataset of the Clock Angle illusion with smooth transitions for points p1 and p2.

    Parameters:
    path (str): Directory to save the generated videos and labels.
    size (int): Number of videos to generate.
    positive_ratio (float): Ratio of positive samples in the dataset.
    video_frames (int): Number of frames in each video.
    framerate (int): Frame rate of the generated videos.

    Returns:
    None
    """
    def limit(x, min_val, max_val):
        x = np.max((x, min_val))
        x = np.min((x, max_val))
        return x

    def get_angle(v1, v2):
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    angleLimit = 0.1

    # DataFrame for storing labels and metadata
    label_df = pd.DataFrame(columns=["name", "label", "start_p1", "end_p1", "start_p2", "end_p2", "angle", "x1", "y1", "x2", "y2", "x3", "y3"])

    for i in tqdm(range(size)):
        # Initial random points p1 and p2, and their end positions for interpolation
        start_p1 = np.random.rand(2) * 64 - 32
        end_p1 = np.random.rand(2) * 64 - 32
        start_p2 = np.random.rand(2) * 64 - 32
        end_p2 = np.random.rand(2) * 64 - 32
        
        # Create video path
        video_path = os.path.join(path, f'clock_angle_video_{i}.mp4')

        # Prepare video writer with the specified framerate
        video_writer = imageio.get_writer(video_path, fps=framerate)

        for frame in range(video_frames):
            interp_factor = frame / (video_frames - 1)
            
            # Interpolate points p1 and p2
            p1 = interpolate_states(start_p1, end_p1, interp_factor)
            p2 = interpolate_states(start_p2, end_p2, interp_factor)

            # Generate point p3 based on positive_ratio and some transformations
            if np.random.rand() > positive_ratio:
                p3 = np.random.rand(2) * 64 - 32
                theta = np.random.rand() * 2 * angleLimit - angleLimit
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                p3 = R @ (p2 - p1) * np.random.rand() + p2 
                xl = limit(p3[0], -32, 32) / p3[0]
                yl = limit(p3[1], -32, 32) / p3[1]
                p3 = R @ (p2 - p1) * np.min((xl, yl)) + p2
                label = 0
            else:
                p3 = (p2 - p1) * np.random.rand() + p2 
                xl = limit(p3[0], -32, 32) / p3[0]
                yl = limit(p3[1], -32, 32) / p3[1]
                p3 = (p2 - p1) * np.min((xl, yl)) + p2
                label = 1

            P = np.concatenate([p1[None, :], p2[None, :], p3[None, :]], axis=0)
            fig = plt.figure(figsize=(4, 4), facecolor='white', dpi=128)
            ax1 = fig.add_subplot(111, aspect='equal')
            ax1.set_xlim([-32, 32])
            ax1.set_ylim([-32, 32])
            ax1.set_axis_off()
            ax1.plot(P[:, 0], P[:, 1], linewidth=1, c='black')

            # Convert the current figure to an image and add it to the video
            fig.canvas.draw()
            img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            video_writer.append_data(img_np)

            plt.close(fig)  # Close the figure after each frame to free up memory

        video_writer.close()

        # Calculate the angle between p1-p2 and p2-p3
        angle = get_angle(p2 - p1, p3 - p2)

        # Save metadata for the video
        label_df.loc[len(label_df)] = [f'clock_angle_video_{i}.mp4', label, start_p1.tolist(), end_p1.tolist(), start_p2.tolist(), end_p2.tolist(), angle, *p1, *p2, *p3]

    # Save labels to CSV
    label_df.to_csv(os.path.join(path, "video_label.csv"), index=False)
