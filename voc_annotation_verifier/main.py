import sys
import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QInputDialog, QScrollArea
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import Qt, QSize

class AnnotationVerifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Verifier")
        self.setGeometry(100, 100, 1280, 720)

        self.dataset_path = ""
        self.current_subset = None
        self.current_image = None
        self.current_xml = None
        self.history = []
        self.removed_samples = []
        self.review_mode = False
        self.review_index = 0
        self.total_samples = 0
        self.processed_samples = 0
        self.kept_samples = 0
        self.state_file = ""

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll_area.setWidget(self.image_label)
        layout.addWidget(scroll_area)

        button_layout = QHBoxLayout()
        self.keep_button = QPushButton("Keep (K)")
        self.remove_button = QPushButton("Remove (R)")
        self.next_button = QPushButton("Next (N)")
        self.prev_button = QPushButton("Previous (P)")
        self.review_button = QPushButton("Toggle Review (V)")
        self.save_button = QPushButton("Save State (S)")

        button_layout.addWidget(self.keep_button)
        button_layout.addWidget(self.remove_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.review_button)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        central_widget.setLayout(layout)

        self.keep_button.clicked.connect(self.keep_sample)
        self.remove_button.clicked.connect(self.remove_sample)
        self.next_button.clicked.connect(self.load_next_sample)
        self.prev_button.clicked.connect(self.load_previous_sample)
        self.review_button.clicked.connect(self.toggle_review_mode)
        self.save_button.clicked.connect(self.save_state)

        # Keyboard shortcuts
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == event.KeyPress:
            key = event.key()
            if key == Qt.Key_K:
                self.keep_sample()
                return True
            elif key == Qt.Key_R:
                self.remove_sample()
                return True
            elif key == Qt.Key_N:
                self.load_next_sample()
                return True
            elif key == Qt.Key_P:
                self.load_previous_sample()
                return True
            elif key == Qt.Key_V:
                self.toggle_review_mode()
                return True
            elif key == Qt.Key_S:
                self.save_state()
                return True
        return super().eventFilter(obj, event)

    def open_dataset(self):
        self.dataset_path = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if self.dataset_path:
            self.state_file = os.path.join(self.dataset_path, 'verifier_state.json')
            choice, _ = QInputDialog.getItem(self, "Start Option", "Choose an option:", ["Start from the beginning", "Resume from last session"], 0, False)
            if choice == "Resume from last session" and os.path.exists(self.state_file):
                self.load_state()
            else:
                self.initialize_state()
            self.load_next_sample()

    def count_samples(self):
        total = 0
        for subset in ['train', 'validation']:
            subset_dir = os.path.join(self.dataset_path, subset)
            total += len([f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return total

    def load_state(self):
        with open(self.state_file, 'r') as f:
            state = json.load(f)
        self.history = state['history']
        self.processed_samples = state['processed_samples']
        self.kept_samples = state['kept_samples']
        self.removed_samples = self.get_defective_samples()
        self.total_samples = self.count_samples()
        QMessageBox.information(self, "State Loaded", f"Loaded state: Processed {self.processed_samples} samples, Kept {self.kept_samples}, Removed {len(self.removed_samples)}")

    def initialize_state(self):
        self.history = []
        self.processed_samples = 0
        self.kept_samples = 0
        self.removed_samples = self.get_defective_samples()
        self.total_samples = self.count_samples()

    def get_defective_samples(self):
        defective_samples = []
        defective_dir = os.path.join(self.dataset_path, 'defective')
        if os.path.exists(defective_dir):
            for subset in ['train', 'validation']:
                subset_dir = os.path.join(defective_dir, subset)
                if os.path.exists(subset_dir):
                    for img in os.listdir(subset_dir):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                            defective_samples.append((subset, img))
        return defective_samples

    def load_next_sample(self):
        if self.review_mode:
            if self.review_index < len(self.removed_samples):
                self.current_subset, self.current_image = self.removed_samples[self.review_index]
                self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
                self.review_index += 1
                self.visualize_sample(review=True)
            else:
                self.review_mode = False
                self.review_index = 0
                QMessageBox.information(self, "Review Completed", "Review mode completed. Returning to normal mode.")
                self.load_next_sample()
        else:
            subsets = ['train', 'validation']
            if not self.current_subset or random.random() < 0.5:
                self.current_subset = random.choice(subsets)
            
            subset_dir = os.path.join(self.dataset_path, self.current_subset)
            image_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                QMessageBox.warning(self, "No Images", f"No more images in {self.current_subset} set.")
                return

            self.current_image = random.choice(image_files)
            self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
            self.history.append((self.current_subset, self.current_image))
            self.processed_samples += 1
            self.visualize_sample()

        self.update_status()

    def visualize_sample(self, review=False):
        if review:
            image_path = os.path.join(self.dataset_path, 'defective', self.current_subset, self.current_image)
            xml_path = os.path.join(self.dataset_path, 'defective', self.current_subset, self.current_xml)
        else:
            image_path = os.path.join(self.dataset_path, self.current_subset, self.current_image)
            xml_path = os.path.join(self.dataset_path, self.current_subset, self.current_xml)

        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", f"Image not found: {image_path}")
            return False

        image = cv2.imread(image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to load image: {image_path}")
            return False

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                label = obj.find('name').text
                
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                self.put_text_with_background(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), (0, 0, 255))
        except ET.ParseError:
            QMessageBox.warning(self, "Error", f"Failed to parse XML: {xml_path}")

        status = "REVIEW MODE: " if review else ""
        self.put_text_with_background(image, f"{status}{self.current_subset}: {self.current_image}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))

        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)

        # Scale the image to fit the window while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        return True

    def put_text_with_background(self, img, text, position, font, font_scale, text_color, bg_color):
        text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
        text_w, text_h = text_size
        x, y = position
        cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, 1, cv2.LINE_AA)

    def keep_sample(self):
        if self.review_mode:
            sample = (self.current_subset, self.current_image)
            if sample in self.removed_samples:
                src_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
                dst_dir = os.path.join(self.dataset_path, self.current_subset)
                if self.move_sample(src_dir, dst_dir):
                    self.removed_samples.remove(sample)
                    self.kept_samples += 1
                    QMessageBox.information(self, "Sample Restored", f"Restored sample: {self.current_image}")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to restore sample: {self.current_image}")
            else:
                QMessageBox.warning(self, "Error", f"Sample {self.current_image} is not in the removed list. Skipping restoration.")
        else:
            self.kept_samples += 1
        self.load_next_sample()

    def remove_sample(self):
        if not self.review_mode:
            src_dir = os.path.join(self.dataset_path, self.current_subset)
            dst_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
            if self.move_sample(src_dir, dst_dir):
                self.removed_samples.append((self.current_subset, self.current_image))
                QMessageBox.information(self, "Sample Removed", f"Removed sample: {self.current_image}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to remove sample: {self.current_image}")
        self.load_next_sample()

    def move_sample(self, src_dir, dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        
        image_src = os.path.join(src_dir, self.current_image)
        xml_src = os.path.join(src_dir, self.current_xml)
        
        image_dst = os.path.join(dst_dir, self.current_image)
        xml_dst = os.path.join(dst_dir, self.current_xml)
        
        try:
            shutil.move(image_src, image_dst)
            shutil.move(xml_src, xml_dst)
            return True
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error moving files: {e}")
            return False

    def load_previous_sample(self):
        if self.review_mode:
            if self.review_index > 1:
                self.review_index -= 2
                self.load_next_sample()
            else:
                QMessageBox.information(self, "Review Mode", "No previous sample in review mode.")
        elif len(self.history) > 1:
            self.history.pop()  # Remove current sample
            self.current_subset, self.current_image = self.history[-1]
            self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
            self.processed_samples -= 1
            self.visualize_sample()
            self.update_status()
        else:
            QMessageBox.information(self, "Navigation", "No previous sample available.")

    def toggle_review_mode(self):
        if self.review_mode:
            self.review_mode = False
            self.review_index = 0
            QMessageBox.information(self, "Review Mode", "Exiting review mode.")
            self.load_next_sample()
        elif self.removed_samples:
            self.review_mode = True
            self.review_index = 0
            QMessageBox.information(self, "Review Mode", "Entering review mode.")
            self.load_next_sample()
        else:
            QMessageBox.information(self, "Review Mode", "No removed samples to review.")

    def update_status(self):
        status = f"Total: {self.total_samples}, Processed: {self.processed_samples}, Kept: {self.kept_samples}, Removed: {len(self.removed_samples)}"
        self.status_label.setText(status)

    def save_state(self):
        state = {
            'history': self.history,
            'processed_samples': self.processed_samples,
            'kept_samples': self.kept_samples
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
        QMessageBox.information(self, "State Saved", "Annotation verifier state has been saved.")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.save_state()
            event.accept()
        else:
            event.ignore()

def main():
    app = QApplication(sys.argv)
    verifier = AnnotationVerifier()
    verifier.show()
    verifier.open_dataset()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()