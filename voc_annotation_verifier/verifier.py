import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil
import sys
import json

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
        self.state_file = os.path.join(dataset_path, 'verifier_state.json')
        self.load_state()

    def count_samples(self):
        total = 0
        for subset in ['train', 'validation']:
            subset_dir = os.path.join(self.dataset_path, subset)
            total += len([f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return total

    def save_state(self):
        state = {
            'history': self.history,
            'removed_samples': self.removed_samples,
            'processed_samples': self.processed_samples,
            'kept_samples': self.kept_samples
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
        print("State saved.")

    def initialize_current_sample(self):
        subsets = ['train', 'validation']
        self.current_subset = random.choice(subsets)
        subset_dir = os.path.join(self.dataset_path, self.current_subset)
        image_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            self.current_image = random.choice(image_files)
            self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
            self.history.append((self.current_subset, self.current_image))
        else:
            print(f"No images found in {self.current_subset} set.")

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            self.history = state['history']
            self.processed_samples = state['processed_samples']
            self.kept_samples = state['kept_samples']
            
            # Sync removed_samples with actual content of defective directory
            self.removed_samples = self.get_defective_samples()
            
            print(f"Loaded state: Processed {self.processed_samples} samples, Kept {self.kept_samples}, Removed {len(self.removed_samples)}")
        else:
            print("No previous state found. Starting from the beginning.")
        
        if not self.current_subset:
            self.initialize_current_sample()

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

    def keep_sample(self):
        if self.review_mode:
            sample = (self.current_subset, self.current_image)
            if sample in self.removed_samples:
                src_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
                dst_dir = os.path.join(self.dataset_path, self.current_subset)
                if self.move_sample(src_dir, dst_dir):
                    self.removed_samples.remove(sample)
                    self.kept_samples += 1
                    print(f"Restored sample: {self.current_image}")
                else:
                    print(f"Failed to restore sample: {self.current_image}")
            else:
                print(f"Sample {self.current_image} is not in the removed list. Skipping restoration.")
        else:
            self.kept_samples += 1
            print(f"Keeping sample: {self.current_image}")

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
                print("Review completed. Returning to normal mode.")
                return self.load_next_sample()

        subsets = ['train', 'validation']
        if not self.current_subset or random.random() < 0.5:
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

    def toggle_review_mode(self):
        if self.review_mode:
            self.review_mode = False
            self.review_index = 0
            print("Exiting review mode.")
            return self.load_next_sample()
        elif self.removed_samples:
            self.review_mode = True
            self.review_index = 0
            print("Entering review mode.")
            return self.load_next_sample()
        else:
            print("No removed samples to review.")
            return True

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
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return False

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
        
        instructions = "Press 'k' to keep, 'r' to remove, 'n' for next, 'p' for previous, 'v' to toggle review mode, 's' to save state, 'q' to quit"
        self.put_text_with_background(image, instructions, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
        
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



def run_annotation_verifier(dataset_path):
    verifier = AnnotationVerifier(dataset_path)
    
    if not verifier.visualize_sample():
        print("Failed to load initial sample. Exiting.")
        return

    while True:
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            verifier.save_state()
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
            if not verifier.toggle_review_mode():
                break
        elif key == ord('s'):
            verifier.save_state()
    
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: voc-annotation-verifier <path_to_dataset>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    print("1. Start from the beginning")
    print("2. Resume from last session")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '2':
        if not os.path.exists(os.path.join(dataset_path, 'verifier_state.json')):
            print("No previous session found. Starting from the beginning.")
    
    run_annotation_verifier(dataset_path)

if __name__ == "__main__":
    main()