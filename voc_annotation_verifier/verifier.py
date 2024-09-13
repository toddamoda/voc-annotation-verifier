import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import sys

class AnnotationVerifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.current_subset = None
        self.current_image = None
        self.current_xml = None
        self.window_name = "Annotation Verifier"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        self.history = []
        self.removed_samples = []
        self.review_mode = False
        self.review_index = 0
        self.total_samples = self.count_samples()
        self.processed_samples = 0
        self.kept_samples = 0
        self.load_next_sample()

    def count_samples(self):
        total = 0
        for subset in ['train', 'validation']:
            subset_dir = os.path.join(self.dataset_path, subset)
            total += len([f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return total

    def load_next_sample(self):
        if self.review_mode:
            if self.review_index < len(self.removed_samples):
                self.current_subset, self.current_image = self.removed_samples[self.review_index]
                self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
                self.review_index += 1
                return self.visualize_sample(review=True)
            else:
                self.review_mode = False
                self.review_index = 0
                return self.load_next_sample()

        subsets = ['train', 'validation']
        if not self.history or random.random() < 0.5:
            self.current_subset = random.choice(subsets)
        
        subset_dir = os.path.join(self.dataset_path, self.current_subset)
        image_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No more images in {self.current_subset} set.")
            return False
        
        self.current_image = random.choice(image_files)
        self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
        self.history.append((self.current_subset, self.current_image))
        self.processed_samples += 1
        
        return self.visualize_sample()

    def load_previous_sample(self):
        if self.review_mode:
            if self.review_index > 1:
                self.review_index -= 2
                return self.load_next_sample()
            else:
                print("No previous sample in review mode.")
                return True
        elif len(self.history) > 1:
            self.history.pop()  # Remove current sample
            self.current_subset, self.current_image = self.history[-1]
            self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
            self.processed_samples -= 1
            return self.visualize_sample()
        else:
            print("No previous sample available.")
            return True

    def visualize_sample(self, review=False):
        if review:
            image_path = os.path.join(self.dataset_path, 'defective', self.current_subset, self.current_image)
            xml_path = os.path.join(self.dataset_path, 'defective', self.current_subset, self.current_xml)
        else:
            image_path = os.path.join(self.dataset_path, self.current_subset, self.current_image)
            xml_path = os.path.join(self.dataset_path, self.current_subset, self.current_xml)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
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
            print(f"Failed to parse XML: {xml_path}")
        
        status = "REVIEW MODE: " if review else ""
        self.put_text_with_background(image, f"{status}{self.current_subset}: {self.current_image}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), (0, 0, 0))
        
        instructions = "Press 'k' to keep, 'r' to remove, 'n' for next, 'p' for previous, 'v' to review removed, 'q' to quit"
        self.put_text_with_background(image, instructions, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
        
        # Add progress information
        progress_text = f"Total: {self.total_samples}, Processed: {self.processed_samples}, Kept: {self.kept_samples}, Removed: {len(self.removed_samples)}"
        self.put_text_with_background(image, progress_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
        
        cv2.imshow(self.window_name, image)
        return True

    def put_text_with_background(self, img, text, position, font, font_scale, text_color, bg_color):
        text_size, _ = cv2.getTextSize(text, font, font_scale, 1)
        text_w, text_h = text_size
        x, y = position
        cv2.rectangle(img, (x, y - text_h - 5), (x + text_w, y + 5), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, 1, cv2.LINE_AA)

    def keep_sample(self):
        if self.review_mode:
            src_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
            dst_dir = os.path.join(self.dataset_path, self.current_subset)
            if self.move_sample(src_dir, dst_dir):
                self.removed_samples.remove((self.current_subset, self.current_image))
                self.kept_samples += 1
                print(f"Restored sample: {self.current_image}")
            else:
                print(f"Failed to restore sample: {self.current_image}")
        else:
            self.kept_samples += 1
            print(f"Keeping sample: {self.current_image}")

    def remove_sample(self):
        if not self.review_mode:
            src_dir = os.path.join(self.dataset_path, self.current_subset)
            dst_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
            if self.move_sample(src_dir, dst_dir):
                self.removed_samples.append((self.current_subset, self.current_image))
                print(f"Removed sample: {self.current_image}")
            else:
                print(f"Failed to remove sample: {self.current_image}")

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
            print(f"Error moving files: {e}")
            return False

    def start_review_mode(self):
        if self.removed_samples:
            self.review_mode = True
            self.review_index = 0
            return self.load_next_sample()
        else:
            print("No removed samples to review.")
            return True

def run_annotation_verifier(dataset_path):
    verifier = AnnotationVerifier(dataset_path)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('k'):
            verifier.keep_sample()
            if not verifier.load_next_sample():
                break
        elif key == ord('r'):
            verifier.remove_sample()
            if not verifier.load_next_sample():
                break
        elif key == ord('n'):
            if not verifier.load_next_sample():
                break
        elif key == ord('p'):
            verifier.load_previous_sample()
        elif key == ord('v'):
            verifier.start_review_mode()
    
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: voc-annotation-verifier <path_to_dataset>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    run_annotation_verifier(dataset_path)

if __name__ == "__main__":
    main()