import sys
import numpy as np
import torch
import cv2
import os
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QListWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from segment_anything import SamPredictor, sam_model_registry

class ImageSegmentationApp(QWidget):
    def __init__(self, src_dir, save_dir, model, predictor):
        super().__init__()
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.images = os.listdir(src_dir)
        self.current_image_index = 0
        self.model = model
        self.predictor = predictor
        self.input_points = []
        self.input_labels = []
        self.mask = None
        self.scale_factor = 1.0  # Scale for zooming

        # Set up the UI components
        self.setWindowTitle("Image Segmentation with SAM")

        # Image list
        self.image_list = QListWidget(self)
        self.image_list.addItems(self.images)
        self.image_list.clicked.connect(self.select_image)

        # Image label
        self.image_label = QLabel(self)

        # Save button
        self.save_button = QPushButton("Save Mask", self)
        self.save_button.clicked.connect(self.save_mask)

        # Layout
        layout = QHBoxLayout()
        image_layout = QVBoxLayout()
        image_layout.addWidget(self.image_label)
        image_layout.addWidget(self.save_button)

        layout.addWidget(self.image_list)
        layout.addLayout(image_layout)

        self.setLayout(layout)

        self.load_image(self.current_image_index)

    def load_image(self, index):
        img_path = os.path.join(self.src_dir, self.images[index])
        self.image = cv2.imread(img_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.update_image_display()

    def update_image_display(self):
        # Display the image with mask and points if available
        display_image = self.image.copy()

        # Draw mask
        if self.mask is not None:
            mask_rgb = cv2.cvtColor(self.mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
            display_image = cv2.addWeighted(display_image, 0.6, mask_rgb, 0.4, 0)

        # Draw points
        for (point, label) in zip(self.input_points, self.input_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(display_image, tuple(point), 5, color, -1)

        # Apply scaling for zooming
        height, width = display_image.shape[:2]
        scaled_image = cv2.resize(display_image, (int(width * self.scale_factor), int(height * self.scale_factor)))

        # Display the image in the label
        q_image = QImage(scaled_image.data, scaled_image.shape[1], scaled_image.shape[0],
                         scaled_image.strides[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def select_image(self):
        # Load the selected image from the list
        selected_item = self.image_list.currentRow()
        self.current_image_index = selected_item
        self.input_points = []
        self.input_labels = []
        self.mask = None
        self.load_image(self.current_image_index)

    def mousePressEvent(self, event):
        # Check if the mouse click happened inside the image label
        if self.image_label.underMouse():
            # Convert the click position to the image label's local coordinates
            label_pos = self.image_label.mapFromGlobal(event.globalPos())
            label_x, label_y = label_pos.x(), label_pos.y()

            # Calculate the original image coordinates based on the scale factor
            x = int(label_x / self.scale_factor)
            y = int(label_y / self.scale_factor)

            # Add point based on mouse button clicked
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:  # Ensure the click is within the image bounds
                if event.button() == Qt.LeftButton:
                    self.input_points.append([x, y])
                    self.input_labels.append(1)  # Left click for label=1 (positive point)
                elif event.button() == Qt.RightButton:
                    self.input_points.append([x, y])
                    self.input_labels.append(0)  # Right click for label=0 (negative point)
                self.run_sam()

    def run_sam(self):
        input_point = np.array(self.input_points)
        input_label = np.array(self.input_labels)
        self.predictor.set_image(self.image)

        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Save the generated mask for further use
        self.mask = masks[0]
        self.update_image_display()

    def save_mask(self):
        # save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        save_dir = self.save_dir
        if save_dir:
            # Save mask as a binary (2D) image, not the combined image
            mask_filename = os.path.join(save_dir, f'mask_{self.images[self.current_image_index]}')
            binary_mask = (self.mask * 255).astype(np.uint8)  # Convert mask to 8-bit format
            cv2.imwrite(mask_filename, binary_mask)

def main():
    sam_checkpoint = "D:/GithubProjects/segment-anything-main/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    app = QApplication(sys.argv)
    src_dir = './dataset/dataset_jpgs_new/t2_pics/train/0'
    save_dir = './dataset/dataset_jpgs_new/t2_msks/train/0'
    os.makedirs(save_dir, exist_ok=True)
    window = ImageSegmentationApp(src_dir, save_dir, sam, predictor)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
