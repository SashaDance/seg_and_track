import os
import cv2
import numpy as np
import json
import torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from services_api.seg_and_track import SegmentorResponse
import cv2.aruco as aruco
from .masks import get_masks_in_rois, get_masks_rois, scale_image
from .conversation

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "tests", "weights", "best_fisheye.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "tests", "data", "output_images")

class Segmentor:
    def __init__(self):
        # Загружаем предобученную модель YOLO
        self.model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            self.model.to('cuda')  # Используем GPU, если доступен
        self.request_counter = 0
        self.palette = {
            0: (7, 7, 132),
            1: (158, 18, 6),
            2: (96, 12, 107),
            3: (112, 82, 0)
        }
        
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL) #DICT_5X5_1000)
        self.aruco_params = aruco.DetectorParameters()

    def segment_image(self, image_file: UploadFile) -> SegmentorResponse:
        self.request_counter += 1
        img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        results = self.model(img)[0]
        
        conf = results.boxes.conf.cpu().numpy().astype(np.float32).tolist()
        class_ids = results.boxes.cls.cpu().numpy().astype(np.uint8).tolist()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.uint32).tolist()

        masks = results.masks
        height, width = results.orig_shape

        mask_objects = [
            {
                "width": mask.shape[1],
                "height": mask.shape[0],
                "mask": mask
            }
            for mask in masks
        ]

        box_objects = [
            {
                "x_min": int(b[0]), 
                "y_min": int(b[1]),
                "x_max": int(b[2]),
                "y_max": int(b[3]),
            }
            for b in boxes
        ]



        if masks is None:
            masks = np.array([])
            scaled_masks = np.empty((0, *(height, width)), dtype=np.uint8)
        else:
            masks = masks.data.cpu().numpy().astype(np.uint8)
            mask_height, mask_width = masks.shape[1:]
            masks = masks.transpose(1, 2, 0)
            scaled_masks = scale_image((mask_height, mask_width), masks, (height, width))
            scaled_masks = scaled_masks.transpose(2, 0, 1)

        rois = get_masks_rois(scaled_masks)
        masks_in_rois = get_masks_in_rois(scaled_masks, rois)

        marker_ids = []
        count_num = 0
        for i, roi in enumerate(rois):
            x = int(roi[1].start)
            y = int(roi[0].start)
            w = int(roi[1].stop - roi[1].start)
            h = int(roi[0].stop - roi[0].start)
            roi_image = img[y:y+h, x:x+w]
            
            detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(roi_image)
            
            if ids is not None: 
                aruco_mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
                pts = corners[0][0].astype(np.int32)
                cv2.fillPoly(aruco_mask, [pts], 1)

                if np.any(masks_in_rois[i] * aruco_mask):
                    marker_ids.append(ids[0][0])
                else:
                    marker_ids.append(count_num)
                    count_num += 1
            else:
                marker_ids.append(count_num)
                count_num += 1

        unique_ids = set(marker_ids)
        for marker_id in unique_ids:
            indices = [i for i, id_ in enumerate(marker_ids) if id_ == marker_id]
            if len(indices) > 1:
                areas = [cv2.contourArea(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]) 
                        for i, mask in enumerate(masks_in_rois) if i in indices]
                max_area_index = indices[np.argmax(areas)]
                for i in indices:
                    if i != max_area_index:
                        marker_ids.pop(i)
                        conf.pop(i)
                        class_ids.pop(i)
                        boxes.pop(i)
                        np.delete(masks_in_rois, i, axis=0)
                        np.delete(rois, i, axis=0)

        num = len(conf)
        if len(marker_ids) > num:
            marker_ids = marker_ids[:num]

        segmentation_objects_msg = to_objects_msg(conf, class_ids, marker_ids, boxes, masks_in_rois, rois, width, height)
    


        img_with_masks = img.copy()
        for mask_data, class_id in zip(mask_objects, class_ids):
            mask = mask_data["mask"]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            color = self.palette[class_id % len(self.palette)]
            color_mask = np.zeros_like(img_with_masks, dtype=np.uint8)
            color_mask[mask_resized == 1] = color
            alpha = 0.5
            img_with_masks = np.where(mask_resized[..., None] == 1, 
                                    cv2.addWeighted(img_with_masks, 1 - alpha, color_mask, alpha, 0), 
                                    img_with_masks)

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        output_image_path = os.path.join(OUTPUT_DIR, f"segmented_image_{self.request_counter}.png")
        cv2.imwrite(output_image_path, img_with_masks)

        shelf_analysis = self.analyze_shelf(class_ids, boxes)

        response = SegmentorResponse(
            num=len(box_objects),
            scores=conf,
            classes_ids=class_ids,
            tracking_ids=marker_ids,
            boxes=box_objects,
            analysis=shelf_analysis
        )

        json_output_path = os.path.join(OUTPUT_DIR, f"segmentation_result_{self.request_counter}.json")
        with open(json_output_path, 'w') as json_file:
            json.dump(response.dict(), json_file, ensure_ascii=False, indent=4)

        return response


    def analyze_shelf(self, class_ids, boxes):
        # Анализ состояния полки
        objects = [{'type': self.get_object_type(class_id), 'position': (int((b[0] + b[2]) / 2), int(b[3]))} for class_id, b in zip(class_ids, boxes)]
        message = {
            'box_on_box': False,
            'shelf_status': ''
        }

        if not objects:
            message['shelf_status'] = 'На полке ничего нет.'
        else:
            shelves = [obj for obj in objects if obj['type'] == 'shelf']
            if not shelves:
                message['shelf_status'] = 'Полка не найдена, невозможно определить положение объектов.'
                return message

            shelf_bottom = max(shelf['position'][1] for shelf in shelves)
            objects_on_shelf = [obj for obj in objects if obj['type'] in ['box', 'container'] and obj['position'][1] <= shelf_bottom]

            if self.is_box_on_box(objects_on_shelf):
                message['box_on_box'] = True
                message['shelf_status'] += 'Коробка находится на коробке. '

            has_box = any(obj['type'] == 'box' for obj in objects_on_shelf)
            has_container = any(obj['type'] == 'container' for obj in objects_on_shelf)

            if has_box and has_container:
                message['shelf_status'] += 'На полке есть коробка и контейнер.'
            elif has_box:
                message['shelf_status'] += 'На полке есть только коробка.'
            elif has_container:
                message['shelf_status'] += 'На полке есть только контейнер.'
            else:
                message['shelf_status'] += 'Коробки или контейнеры отсутствуют на полке.'

        return message

    def get_object_type(self, class_id):
        if class_id == 0:
            return 'box'
        elif class_id == 1:
            return 'container'
        elif class_id == 2:
            return 'person'
        elif class_id == 3:
            return 'shelf'
        else:
            return 'unknown'

    def is_box_on_box(self, objects):
        for obj1 in objects:
            for obj2 in objects:
                if obj1 != obj2 and obj1['type'] == 'box' and obj2['type'] == 'box':
                    height_obj1 = obj1['position'][0] - obj1['position'][1]  # разница по оси y
                    height_obj2 = obj2['position'][0] - obj2['position'][1]
                    vertical_distance = abs(obj1['position'][1] - obj2['position'][1])

                    if obj1['position'][1] > obj2['position'][1] and vertical_distance <= max(height_obj1, height_obj2):
                        return True
        return False
    

    def to_objects_msg(scores, classes_ids, tracking_ids, boxes,
        masks_in_rois, rois, widths, heights):
        num = len(scores)

        if isinstance(widths, Number):
            widths = [widths] * num
        if isinstance(heights, Number):
            heights = [heights] * num


        objects_msg = Objects()
        # objects_msg.header = header
        objects_msg.num = num
        objects_msg.scores.extend(scores)
        objects_msg.classes_ids.extend(classes_ids)
        objects_msg.tracking_ids.extend(tracking_ids)
        objects_msg.boxes.extend(map(to_box_msg, boxes))
        objects_msg.masks.extend(map(to_mask_msg, masks_in_rois, rois, widths, heights))

        return objects_msg
    
    
