#!/usr/bin/env python3

import os
import rospy
import sys
import sys
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Add the parent directory (object_detection) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from odcl_lib import Localisation, inference

def load_model(model_path, num_classes=16):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_normalized_bboxes(prediction, image_size, threshold=0.5):
    boxes = prediction['boxes'].detach().cpu().numpy()
    scores = prediction['scores'].detach().cpu().numpy()
    h, w = image_size
    norm_bboxes = []
    for box, score in zip(boxes, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_width = (xmax - xmin) / w
            box_height = (ymax - ymin) / h
            norm_bboxes.append((x_center, y_center, box_width, box_height))
    return norm_bboxes

def main():
    rospy.init_node('inference_node')
    image_folder = os.path.expanduser("~/suas_final/src/object_detection/aerial_images")
    output_file = os.path.expanduser("~/suas_final/src/object_detection/lat_long.txt")
    model_path = os.path.expanduser("~/suas_final/src/object_detection/models/fasterrcnn_stage3_epoch26.pth")

    global device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    transform = T.ToTensor()
    model = inference.get_model()


    with open(output_file, 'w') as f_out:
        for img_file in os.listdir(image_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(image_folder, img_file)
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).unsqueeze(0).to(device)

                prediction = model(image_tensor)[0]
                norm_bboxes = extract_normalized_bboxes(prediction, image.size[::-1])

                gps = Localisation.extract_gps(img_path)
                if gps:
                    center_lat, center_lon = gps
                    bbox_coords = Localisation.yolo_bbox_center_to_latlon(
                        norm_bboxes, image.width, image.height, center_lat, center_lon,
                        altitude_meters=75 * 0.3048, fov_horizontal=90, fov_vertical=60
                    )

                    for lat, lon in bbox_coords:
                        f_out.write(f"{img_file}: {lat:.8f}, {lon:.8f}\n")
                        rospy.loginfo(f"{img_file}: {lat:.8f}, {lon:.8f}")
                else:
                    rospy.logwarn(f"No GPS info found for {img_file}")

    rospy.loginfo("Inference and logging completed.")

if __name__ == "__main__":
    main()
