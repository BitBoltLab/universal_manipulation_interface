#!/usr/bin/env python3
"""
Bimanual Robot Dataset Visualizer
Visualizes dual-arm robot demonstrations with synchronized multi-modal data display.

Features:
- Dual-arm pose visualization (numerical and 3D coordinate frames)
- Gripper width display
- Multi-camera synchronized image display
- Episode navigation with playback controls
- Time-synchronized visualization across all modalities

Usage:
    python visualize_dataset.py <path_to_dataset.zarr.zip>
"""

import sys
import argparse
import numpy as np
import zarr
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QGroupBox, QGridLayout,
    QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from scipy.spatial.transform import Rotation
from OpenGL.GL import *

# Register imagecodecs with numcodecs for JPEG-XL support
try:
    from imagecodecs.numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("Warning: imagecodecs not available, compressed images may not load")
    pass


class DatasetLoader:
    """Load and manage zarr dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.store = zarr.ZipStore(str(self.dataset_path), mode='r')
        self.root = zarr.group(store=self.store)
        
        self.data = self.root['data']
        self.meta = self.root['meta']
        self.episode_ends = self.meta['episode_ends'][:]
        
        # Parse data structure
        self.n_episodes = len(self.episode_ends)
        self.n_robots = self._count_robots()
        self.n_cameras = self._count_cameras()
        
        print(f"Dataset loaded: {self.n_episodes} episodes, {self.n_robots} robots, {self.n_cameras} cameras")
    
    def _count_robots(self):
        """Count number of robots from data keys"""
        robot_keys = [k for k in self.data.keys() if k.startswith('robot') and k.endswith('_eef_pos')]
        return len(robot_keys)
    
    def _count_cameras(self):
        """Count number of cameras from data keys"""
        camera_keys = [k for k in self.data.keys() if k.startswith('camera') and k.endswith('_rgb')]
        return len(camera_keys)
    
    def get_episode_range(self, episode_idx):
        """Get start and end frame indices for an episode"""
        start_idx = 0 if episode_idx == 0 else self.episode_ends[episode_idx - 1]
        end_idx = self.episode_ends[episode_idx]
        return start_idx, end_idx
    
    def get_episode_length(self, episode_idx):
        """Get number of frames in an episode"""
        start_idx, end_idx = self.get_episode_range(episode_idx)
        return end_idx - start_idx
    
    def get_frame_data(self, episode_idx, frame_idx):
        """Get all data for a specific frame in an episode"""
        start_idx, end_idx = self.get_episode_range(episode_idx)
        global_idx = start_idx + frame_idx
        
        if global_idx >= end_idx:
            return None
        
        frame_data = {}
        
        # Load robot data
        for robot_id in range(self.n_robots):
            prefix = f'robot{robot_id}'
            frame_data[f'{prefix}_eef_pos'] = self.data[f'{prefix}_eef_pos'][global_idx]
            frame_data[f'{prefix}_eef_rot_axis_angle'] = self.data[f'{prefix}_eef_rot_axis_angle'][global_idx]
            frame_data[f'{prefix}_gripper_width'] = self.data[f'{prefix}_gripper_width'][global_idx]
        
        # Load camera images
        for cam_id in range(self.n_cameras):
            frame_data[f'camera{cam_id}_rgb'] = self.data[f'camera{cam_id}_rgb'][global_idx]
        
        return frame_data


class CoordinateFrameItem(gl.GLGraphicsItem.GLGraphicsItem):
    """3D coordinate frame visualization (XYZ axes)"""
    
    def __init__(self, size=0.1, line_width=3.0):
        super().__init__()
        self.size = size
        self.line_width = line_width
        self.setGLOptions('additive')
    
    def paint(self):
        self.setupGLState()
        
        # Set line width for thicker axes
        glLineWidth(self.line_width)
        
        # Draw XYZ axes
        glBegin(GL_LINES)
        
        # X axis (red)
        glColor4f(1.0, 0.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(self.size, 0, 0)
        
        # Y axis (green)
        glColor4f(0.0, 1.0, 0.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, self.size, 0)
        
        # Z axis (blue)
        glColor4f(0.0, 0.0, 1.0, 1.0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, self.size)
        
        glEnd()
        
        # Reset line width
        glLineWidth(1.0)


class Pose3DVisualizer(QWidget):
    """3D visualization of robot poses with coordinate frames"""
    
    def __init__(self, n_robots=2):
        super().__init__()
        self.n_robots = n_robots
        self.frames = []
        self.center_point = np.array([0.0, 0.0, 0.0])
        self.auto_scaled = False  # Track if we've done initial auto-scaling
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D view with larger minimum size
        self.view = gl.GLViewWidget()
        self.view.setMinimumSize(600, 500)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setCameraPosition(distance=0.8, elevation=20, azimuth=45)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.scale(0.05, 0.05, 0.05)
        self.view.addItem(grid)
        
        # Add base frame (world origin) - larger and more visible
        self.base_frame = CoordinateFrameItem(size=0.15, line_width=5.0)
        self.view.addItem(self.base_frame)
        
        # Add coordinate frames for each robot
        colors = [(1, 0, 0, 0.5), (0, 0, 1, 0.5)]  # Red for robot0, Blue for robot1
        for i in range(n_robots):
            frame = CoordinateFrameItem(size=0.08, line_width=4.0)
            self.view.addItem(frame)
            self.frames.append(frame)
        
        layout.addWidget(self.view)
        self.setLayout(layout)
    
    def reset_auto_scale(self):
        """Reset auto-scale flag for new episode"""
        self.auto_scaled = False
    
    def update_poses(self, positions, rotations, auto_scale=False):
        """Update robot poses
        
        Args:
            positions: List of (x, y, z) positions for each robot
            rotations: List of axis-angle rotations for each robot
            auto_scale: If True, auto-adjust camera view (only on episode load)
        """
        if len(positions) == 0:
            return
        
        # Only auto-scale once per episode or when explicitly requested
        if auto_scale and not self.auto_scaled:
            # Include both origin (0,0,0) and robot positions for bounding box
            all_positions = np.array(positions)
            origin = np.array([[0.0, 0.0, 0.0]])
            all_points = np.vstack([origin, all_positions])
            
            # Calculate center point including origin
            self.center_point = np.mean(all_points, axis=0)
            
            # Calculate bounding box including origin and robot positions
            min_pos = np.min(all_points, axis=0)
            max_pos = np.max(all_points, axis=0)
            extent = np.linalg.norm(max_pos - min_pos)  # Use diagonal distance
            
            # Auto-adjust camera distance to fit everything in view
            if extent > 0.01:  # Avoid division by zero
                distance = max(0.3, extent * 1.5)  # 1.5x extent for comfortable view
                self.view.setCameraPosition(
                    distance=distance,
                    elevation=20,
                    azimuth=45
                )
                # Set camera center separately
                self.view.opts['center'] = pg.Vector(self.center_point[0], self.center_point[1], self.center_point[2])
            
            self.auto_scaled = True
        
        # Update robot frames
        for i, (pos, rot_aa) in enumerate(zip(positions, rotations)):
            if i >= len(self.frames):
                break
            
            # Convert axis-angle to rotation matrix
            rot = Rotation.from_rotvec(rot_aa)
            rot_mat = rot.as_matrix()
            
            # Create 4x4 transformation matrix
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rot_mat
            transform[:3, 3] = pos
            
            # Apply transformation using setTransform with flattened matrix
            self.frames[i].resetTransform()
            # PyQtGraph expects row-major order
            self.frames[i].setTransform(pg.Transform3D(transform))


class RobotStatePanel(QWidget):
    """Display numerical robot state (position, rotation, gripper)"""
    
    def __init__(self, robot_id=0):
        super().__init__()
        self.robot_id = robot_id
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Robot {robot_id} State")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title)
        
        # Position
        pos_group = QGroupBox("Position (m)")
        pos_layout = QGridLayout()
        self.pos_labels = {}
        for i, axis in enumerate(['X', 'Y', 'Z']):
            pos_layout.addWidget(QLabel(f"{axis}:"), i, 0)
            label = QLabel("0.000")
            label.setFont(QFont("Courier", 10))
            self.pos_labels[axis] = label
            pos_layout.addWidget(label, i, 1)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Rotation (axis-angle)
        rot_group = QGroupBox("Rotation (axis-angle)")
        rot_layout = QGridLayout()
        self.rot_labels = {}
        for i, axis in enumerate(['RX', 'RY', 'RZ']):
            rot_layout.addWidget(QLabel(f"{axis}:"), i, 0)
            label = QLabel("0.000")
            label.setFont(QFont("Courier", 10))
            self.rot_labels[axis] = label
            rot_layout.addWidget(label, i, 1)
        rot_group.setLayout(rot_layout)
        layout.addWidget(rot_group)
        
        # Gripper width
        gripper_group = QGroupBox("Gripper Width (m)")
        gripper_layout = QHBoxLayout()
        self.gripper_label = QLabel("0.000")
        self.gripper_label.setFont(QFont("Courier", 10))
        gripper_layout.addWidget(self.gripper_label)
        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_state(self, position, rotation, gripper_width):
        """Update displayed state values"""
        # Position
        for i, axis in enumerate(['X', 'Y', 'Z']):
            self.pos_labels[axis].setText(f"{position[i]:.3f}")
        
        # Rotation
        for i, axis in enumerate(['RX', 'RY', 'RZ']):
            self.rot_labels[axis].setText(f"{rotation[i]:.3f}")
        
        # Gripper
        self.gripper_label.setText(f"{gripper_width[0]:.3f}")


class CameraPanel(QWidget):
    """Display camera image"""
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title
        title = QLabel(f"Camera {camera_id}")
        title.setFont(QFont("Arial", 10, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Image display with fixed aspect ratio
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setScaledContents(False)  # Don't scale contents, we'll handle it
        layout.addWidget(self.image_label)
        
        self.setLayout(layout)
    
    def update_image(self, image_array):
        """Update displayed image
        
        Args:
            image_array: numpy array (H, W, 3) in RGB format
        """
        if image_array is None:
            return
        
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        
        # Convert to QImage (RGB888)
        q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)


class PlaybackControls(QWidget):
    """Playback control panel with episode navigation"""
    
    def __init__(self, n_episodes=1, max_frames=100):
        super().__init__()
        self.n_episodes = n_episodes
        self.max_frames = max_frames
        self.is_playing = False
        
        layout = QVBoxLayout()
        
        # Episode selection
        episode_layout = QHBoxLayout()
        episode_layout.addWidget(QLabel("Episode:"))
        self.episode_spinbox = QSpinBox()
        self.episode_spinbox.setRange(0, n_episodes - 1)
        self.episode_spinbox.setValue(0)
        episode_layout.addWidget(self.episode_spinbox)
        
        self.episode_info_label = QLabel(f"/ {n_episodes - 1}")
        episode_layout.addWidget(self.episode_info_label)
        episode_layout.addStretch()
        layout.addLayout(episode_layout)
        
        # Frame slider
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, max_frames - 1)
        self.frame_slider.setValue(0)
        frame_layout.addWidget(self.frame_slider)
        
        # Frame counter
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.frame_label)
        layout.addLayout(frame_layout)
        
        # Playback buttons
        button_layout = QHBoxLayout()
        
        self.prev_episode_btn = QPushButton("◀◀ Prev Episode")
        button_layout.addWidget(self.prev_episode_btn)
        
        self.play_pause_btn = QPushButton("▶ Play")
        button_layout.addWidget(self.play_pause_btn)
        
        self.next_episode_btn = QPushButton("Next Episode ▶▶")
        button_layout.addWidget(self.next_episode_btn)
        
        layout.addLayout(button_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(30)  # 30 FPS default
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("30 FPS")
        speed_layout.addWidget(self.speed_label)
        layout.addLayout(speed_layout)
        
        self.setLayout(layout)
    
    def set_max_frames(self, max_frames):
        """Update maximum frame count for current episode"""
        self.max_frames = max_frames
        self.frame_slider.setMaximum(max_frames - 1)
        self.update_frame_label()
    
    def update_frame_label(self):
        """Update frame counter display"""
        current = self.frame_slider.value()
        self.frame_label.setText(f"{current} / {self.max_frames - 1}")
    
    def set_playing(self, playing):
        """Update play/pause button state"""
        self.is_playing = playing
        if playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")


class DatasetVisualizerWindow(QMainWindow):
    """Main visualization window"""
    
    def __init__(self, dataset_path):
        super().__init__()
        
        # Load dataset
        self.dataset = DatasetLoader(dataset_path)
        
        # State
        self.current_episode = 0
        self.current_frame = 0
        self.is_playing = False
        
        # Setup UI
        self.setWindowTitle(f"Dataset Visualizer - {Path(dataset_path).name}")
        self.setGeometry(100, 100, 1600, 900)
        
        self.setup_ui()
        self.setup_timer()
        
        # Load first frame
        self.load_episode(0)
    
    def setup_ui(self):
        """Setup user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Top section: 3D view and robot states
        top_splitter = QSplitter(Qt.Horizontal)
        
        # 3D pose visualization
        self.pose_3d = Pose3DVisualizer(n_robots=self.dataset.n_robots)
        top_splitter.addWidget(self.pose_3d)
        
        # Robot state panels
        robot_states_widget = QWidget()
        robot_states_layout = QHBoxLayout()
        self.robot_panels = []
        for i in range(self.dataset.n_robots):
            panel = RobotStatePanel(robot_id=i)
            self.robot_panels.append(panel)
            robot_states_layout.addWidget(panel)
        robot_states_widget.setLayout(robot_states_layout)
        top_splitter.addWidget(robot_states_widget)
        
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(top_splitter, stretch=3)
        
        # Middle section: Camera views with grid layout for better aspect ratio
        camera_widget = QWidget()
        camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Use grid layout for 1-3 cameras to maintain aspect ratio
        if self.dataset.n_cameras <= 3:
            camera_layout = QHBoxLayout()
            camera_layout.setSpacing(10)
        else:
            camera_layout = QGridLayout()
            camera_layout.setSpacing(10)
        
        self.camera_panels = []
        for i in range(self.dataset.n_cameras):
            panel = CameraPanel(camera_id=i)
            self.camera_panels.append(panel)
            
            if self.dataset.n_cameras <= 3:
                camera_layout.addWidget(panel)
            else:
                # Grid layout for 4+ cameras (2 columns)
                row = i // 2
                col = i % 2
                camera_layout.addWidget(panel, row, col)
        
        camera_widget.setLayout(camera_layout)
        main_layout.addWidget(camera_widget, stretch=2)
        
        # Bottom section: Playback controls
        self.controls = PlaybackControls(
            n_episodes=self.dataset.n_episodes,
            max_frames=100
        )
        
        # Connect signals
        self.controls.episode_spinbox.valueChanged.connect(self.on_episode_changed)
        self.controls.frame_slider.valueChanged.connect(self.on_frame_changed)
        self.controls.play_pause_btn.clicked.connect(self.toggle_playback)
        self.controls.prev_episode_btn.clicked.connect(self.prev_episode)
        self.controls.next_episode_btn.clicked.connect(self.next_episode)
        self.controls.speed_slider.valueChanged.connect(self.on_speed_changed)
        
        main_layout.addWidget(self.controls)
        
        central_widget.setLayout(main_layout)
    
    def setup_timer(self):
        """Setup playback timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_frame)
        self.timer.setInterval(33)  # ~30 FPS
    
    def load_episode(self, episode_idx):
        """Load a specific episode"""
        self.current_episode = episode_idx
        self.current_frame = 0
        
        # Reset auto-scale flag for new episode
        self.pose_3d.reset_auto_scale()
        
        # Update controls
        episode_length = self.dataset.get_episode_length(episode_idx)
        self.controls.set_max_frames(episode_length)
        self.controls.episode_spinbox.setValue(episode_idx)
        self.controls.frame_slider.setValue(0)
        
        # Load first frame with auto-scaling
        self.update_visualization(auto_scale=True)
    
    def update_visualization(self, auto_scale=False):
        """Update all visualizations with current frame data
        
        Args:
            auto_scale: If True, auto-adjust 3D camera view (only on episode load)
        """
        frame_data = self.dataset.get_frame_data(self.current_episode, self.current_frame)
        
        if frame_data is None:
            return
        
        # Update robot poses (3D and numerical)
        positions = []
        rotations = []
        
        for i in range(self.dataset.n_robots):
            prefix = f'robot{i}'
            pos = frame_data[f'{prefix}_eef_pos']
            rot = frame_data[f'{prefix}_eef_rot_axis_angle']
            gripper = frame_data[f'{prefix}_gripper_width']
            
            positions.append(pos)
            rotations.append(rot)
            
            # Update numerical display
            self.robot_panels[i].update_state(pos, rot, gripper)
        
        # Update 3D visualization (auto_scale only on episode load)
        self.pose_3d.update_poses(positions, rotations, auto_scale=auto_scale)
        
        # Update camera images
        for i in range(self.dataset.n_cameras):
            image = frame_data[f'camera{i}_rgb']
            self.camera_panels[i].update_image(image)
        
        # Update frame counter
        self.controls.update_frame_label()
    
    def on_episode_changed(self, episode_idx):
        """Handle episode selection change"""
        if episode_idx != self.current_episode:
            self.load_episode(episode_idx)
    
    def on_frame_changed(self, frame_idx):
        """Handle frame slider change"""
        self.current_frame = frame_idx
        self.update_visualization()
    
    def on_speed_changed(self, speed):
        """Handle playback speed change"""
        fps = speed
        interval = int(1000 / fps)
        self.timer.setInterval(interval)
        self.controls.speed_label.setText(f"{fps} FPS")
    
    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.controls.set_playing(self.is_playing)
        
        if self.is_playing:
            self.timer.start()
        else:
            self.timer.stop()
    
    def advance_frame(self):
        """Advance to next frame during playback"""
        max_frame = self.dataset.get_episode_length(self.current_episode) - 1
        
        if self.current_frame < max_frame:
            self.current_frame += 1
            self.controls.frame_slider.setValue(self.current_frame)
        else:
            # End of episode, stop or loop
            self.is_playing = False
            self.controls.set_playing(False)
            self.timer.stop()
    
    def prev_episode(self):
        """Go to previous episode"""
        if self.current_episode > 0:
            self.load_episode(self.current_episode - 1)
    
    def next_episode(self):
        """Go to next episode"""
        if self.current_episode < self.dataset.n_episodes - 1:
            self.load_episode(self.current_episode + 1)


def main():
    parser = argparse.ArgumentParser(description='Visualize bimanual robot dataset')
    parser.add_argument('dataset_path', type=str, help='Path to dataset.zarr.zip file')
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset_path).exists():
        print(f"Error: Dataset not found at {args.dataset_path}")
        sys.exit(1)
    
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = DatasetVisualizerWindow(args.dataset_path)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
