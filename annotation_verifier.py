import os
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import shutil

class AnnotationVerifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.current_subset = None
        self.current_image = None
        self.current_xml = None
        self.window_name = "Annotation Verifier"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        self.load_next_sample()

    def load_next_sample(self):
        subsets = ['train', 'validation']
        if self.current_subset is None or random.random() < 0.5:
            self.current_subset = random.choice(subsets)
        
        subset_dir = os.path.join(self.dataset_path, self.current_subset)
        image_files = [f for f in os.listdir(subset_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No more images in {self.current_subset} set.")
            return False
        
        self.current_image = random.choice(image_files)
        self.current_xml = os.path.splitext(self.current_image)[0] + '.xml'
        
        return self.visualize_sample()

    def visualize_sample(self):
        image_path = os.path.join(self.dataset_path, self.current_subset, self.current_image)
        xml_path = os.path.join(self.dataset_path, self.current_subset, self.current_xml)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return False
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.putText(image, f"{self.current_subset}: {self.current_image}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, "Press 'k' to keep, 'r' to remove, 'n' for next, 'q' to quit", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, image)
        return True

    def keep_sample(self):
        print(f"Keeping sample: {self.current_image}")

    def remove_sample(self):
        defective_dir = os.path.join(self.dataset_path, 'defective', self.current_subset)
        os.makedirs(defective_dir, exist_ok=True)
        
        image_src = os.path.join(self.dataset_path, self.current_subset, self.current_image)
        xml_src = os.path.join(self.dataset_path, self.current_subset, self.current_xml)
        
        image_dst = os.path.join(defective_dir, self.current_image)
        xml_dst = os.path.join(defective_dir, self.current_xml)
        
        shutil.move(image_src, image_dst)
        shutil.move(xml_src, xml_dst)
        
        print(f"Removed sample: {self.current_image}")

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
    
    cv2.destroyAllWindows()

# if __name__ == "__main__":
#     dataset_path = "Humandata"  # Change this to your dataset path
#     run_annotation_verifier(dataset_path)