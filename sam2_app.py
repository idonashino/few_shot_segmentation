import os
import numpy as np
import torch
import cv2
from PyQt5.QtWidgets import QApplication, QLabel, QSlider, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Input/output directories written directly in the code
input_dir = "./videos/bedroom"  # Folder containing multiple video subfolders
save_dir = "./output_masks"  # Folder to save masks

# SAM-2 model and device initialization
device = torch.device("cuda")
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# App GUI class
class VideoAnnotationApp(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize dictionaries to track points and labels for each frame
        self.points_dict = {}
        self.labels_dict = {}
        self.initUI()

    def initUI(self):
        # Layout
        self.setWindowTitle('SAM-2 Video Annotation App')
        self.setGeometry(100, 100, 800, 600)

        # Video display label
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Slider for navigating video frames
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_frame)

        # Buttons for propagation and saving masks
        self.prop_button = QPushButton('Propagate throughout the video', self)
        self.prop_button.clicked.connect(self.propagate_masks)

        self.save_button = QPushButton('Save Masks', self)
        self.save_button.clicked.connect(self.save_masks)

        # Layout for adding widgets
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        layout.addWidget(self.prop_button)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        # Variables to track state
        self.current_video = None
        self.frames = []
        self.current_frame_idx = 0
        self.points = []
        self.labels = []
        self.masks = {}
        self.inference_state = None

        # Load first video
        self.load_video()

    def load_video(self):
        # Hardcoded input folder
        video_dir = input_dir  # Example subfolder
        self.frames = [os.path.join(video_dir, f) for f in sorted(os.listdir(video_dir)) if f.endswith(".jpg")]
        self.slider.setMaximum(len(self.frames) - 1)
        self.init_inference_state()
        self.update_frame()

    def init_inference_state(self):
        # Initialize SAM-2 inference state for the video
        self.inference_state = predictor.init_state(video_path=os.path.dirname(self.frames[0]))
        predictor.reset_state(self.inference_state)

    def update_frame(self):
        # Load and display current frame
        self.current_frame_idx = self.slider.value()
        frame_path = self.frames[self.current_frame_idx]
        pixmap = QPixmap(frame_path)
        self.label.setPixmap(pixmap)

        # Draw any saved points
        self.update_points()

        # Check if there's a mask for the current frame and display it
        self.display_mask()

    def update_points(self):
        # Retrieve points and labels for the current frame
        self.points = self.points_dict.get(self.current_frame_idx, [])
        self.labels = self.labels_dict.get(self.current_frame_idx, [])

        # Draw points (left: green, right: red) on the current frame
        pixmap = QPixmap(self.frames[self.current_frame_idx])
        self.label.setPixmap(pixmap)
        self.display_mask()

    def display_mask(self):
        # Generate and display the mask for the current frame
        if self.points:
            _, _, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=self.current_frame_idx,
                obj_id=1,  # Assuming single object for simplicity
                points=np.array(self.points, dtype=np.float32),
                labels=np.array(self.labels, dtype=np.int32),
            )
            # Remove singleton dimension from the mask
            mask = (out_mask_logits[0] > 0).cpu().numpy().squeeze()

            # Store mask for current frame
            self.masks[self.current_frame_idx] = mask

            # Load the current image using cv2
            img = cv2.imread(self.frames[self.current_frame_idx])

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert mask to 3-channel grayscale to overlay on the image
            mask_overlay = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

            # Apply the mask overlay on the image (with transparency)
            alpha = 0.5  # Transparency factor
            img_with_mask = cv2.addWeighted(img, 1, mask_overlay, alpha, 0)

            # Convert the OpenCV image to QImage
            h, w, ch = img_with_mask.shape
            bytes_per_line = ch * w
            q_img = QImage(img_with_mask.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Display the image with mask in the QLabel
            self.label.setPixmap(QPixmap.fromImage(q_img))
        # Check if we have a mask for the current frame
        if self.current_frame_idx in self.masks:
            mask = self.masks[self.current_frame_idx]
            
            # Check if the mask is a dictionary (for multiple objects)
            if isinstance(mask, dict):
                # Assuming we want to display the mask for the first object in the dictionary
                obj_id = next(iter(mask))  # Get the first object ID
                mask = mask[obj_id]
            
            # If it's not a dictionary, proceed normally (single object case)
            # Ensure mask is a numpy array
            if isinstance(mask, np.ndarray):
                # Remove singleton dimension from the mask
                mask = mask.squeeze()

                # Load the current image using cv2
                img = cv2.imread(self.frames[self.current_frame_idx])

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Convert mask to 3-channel grayscale to overlay on the image
                mask_overlay = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)

                # Apply the mask overlay on the image (with transparency)
                alpha = 0.5  # Transparency factor
                img_with_mask = cv2.addWeighted(img, 1, mask_overlay, alpha, 0)

                # Convert the OpenCV image to QImage
                h, w, ch = img_with_mask.shape
                bytes_per_line = ch * w
                q_img = QImage(img_with_mask.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # Display the image with mask in the QLabel
                self.label.setPixmap(QPixmap.fromImage(q_img))
            else:
                print("\033[94m" + "Mask is not a valid numpy array." + "\033[0m")
        else:
            print("\033[94m" + "No mask found for the current frame." + "\033[0m")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton or event.button() == Qt.RightButton:
            x, y = event.pos().x(), event.pos().y()
            label = 1 if event.button() == Qt.LeftButton else 0
            
            # Store points and labels for the current frame
            if self.current_frame_idx not in self.points_dict:
                self.points_dict[self.current_frame_idx] = []
                self.labels_dict[self.current_frame_idx] = []

            self.points_dict[self.current_frame_idx].append((x, y))
            self.labels_dict[self.current_frame_idx].append(label)
            self.update_points()

    def propagate_masks(self):
        # Propagate masks throughout the entire video
        propagated_masks = {}  # Temporary dict to store propagated masks
        
        # Loop through each frame and generate the masks
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
            # For each object in the frame, store the mask
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy()  # Get the mask as numpy array
                if out_frame_idx not in propagated_masks:
                    propagated_masks[out_frame_idx] = {}
                propagated_masks[out_frame_idx][obj_id] = mask
        
        # Store the propagated masks into self.masks for later use
        for frame_idx, masks in propagated_masks.items():
            self.masks[frame_idx] = masks

        print("\033[94m" + "Masks are generated successfully!" + "\033[0m")
        # After propagation, update the current frame display
        self.update_frame()

    def save_masks(self):
        # Save masks to the output folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for frame_idx, mask in self.masks.items():
            # If mask is a dictionary (for multiple objects), iterate through it
            if isinstance(mask, dict):
                for obj_id, obj_mask in mask.items():
                    # Ensure mask is a 2D array and uint8 type
                    obj_mask = obj_mask.squeeze()  # Remove singleton dimensions
                    obj_mask = (obj_mask * 255).astype(np.uint8)  # Scale to [0, 255]

                    mask_img = Image.fromarray(obj_mask)
                    mask_img.save(os.path.join(save_dir, f"mask_{frame_idx}_obj_{obj_id}.jpg"))
            else:
                # Single object mask
                mask = mask.squeeze()  # Ensure it's 2D
                mask = (mask * 255).astype(np.uint8)

                mask_img = Image.fromarray(mask)
                mask_img.save(os.path.join(save_dir, f"mask_{frame_idx}.jpg"))

            print("\033[94m" + "Masks are saved successfully!" + "\033[0m")

# Main application loop
app = QApplication([])
window = VideoAnnotationApp()
window.show()
app.exec_()
