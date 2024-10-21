import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2.aruco as aruco
from masks import get_masks_in_rois, get_masks_rois, scale_image, reconstruct_masks
from visualization import draw_objects
from conversions import to_mask_msg
from seg_and_track import SegmentorResponse, Box, Mask, Roi
from DoF_pose_estim_aruco import my_estimatePoseSingleMarkers

class Segmentor:
    def __init__(self):
        # Загружаем модель - переделать на hugginface
        self.model = YOLO("/home/angelika/Desktop/Cogmodel/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/weights/best_fisheye.pt")
        print("Model loaded")
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.request_counter = 0
        self.palette = ((0, 0, 255), (255, 0, 0))

        self.colors_palette = {
            0: (7, 7, 132),
            1: (158, 18, 6),
            2: (96, 12, 107),
            3: (112, 82, 0)
        }

        # Параметры Aruco
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.aruco_params = aruco.DetectorParameters()
        self.camera_matrix = np.array([[800, 0, 640], [0, 800, 480], [0, 0, 1]])  # Стандартные значения
        self.dist_coeffs = np.array([0, 0, 0, 0, 0])  # Отсутствие дисторсий


    def segment_image(self, image_path):
        self.request_counter += 1

        
        img = cv2.imread(image_path)
        h, w = img.shape[:2]

       
        results = self.model(img)[0]

        
        conf = results.boxes.conf.cpu().numpy().astype(np.float32).tolist()
        class_ids = results.boxes.cls.cpu().numpy().astype(np.uint8).tolist()
        boxes = results.boxes.xyxy.cpu().numpy().astype(np.uint32).tolist()
        masks = results.masks if results.masks is not None else None

        if masks is None:
            masks = np.array([])
            scaled_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            masks = masks.data.cpu().numpy().astype(np.uint8)
            mask_height, mask_width = masks.shape[1:]
            masks = masks.transpose(1, 2, 0)
            scaled_masks = scale_image((mask_height, mask_width), masks, (h, w))
            scaled_masks = scaled_masks.transpose(2, 0, 1)

        # Реконструкция масок
        rois = get_masks_rois(scaled_masks)
        masks_in_rois = get_masks_in_rois(scaled_masks, rois)

        # Обработка Aruco маркеров и расчет 6DoF поз
        marker_ids = []
        marker_poses = []
        count_num = 0
        marker_length = 80  # mm

        for i, roi in enumerate(rois):
            x = int(roi[1].start)
            y = int(roi[0].start)
            w_roi = int(roi[1].stop - roi[1].start)
            h_roi = int(roi[0].stop - roi[0].start)
            roi_image = img[y:y + h_roi, x:x + w_roi]
            
            detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(roi_image)
            
            if ids is not None: 
                aruco_mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
                pts = corners[0][0].astype(np.int32)
                cv2.fillPoly(aruco_mask, [pts], 1)

                if np.any(masks_in_rois[i] * aruco_mask):
                    marker_ids.append(ids[0][0])

                    # Рассчет 6DoF позы
                    rvecs, tvecs, _ = my_estimatePoseSingleMarkers(corners, marker_length, self.camera_matrix,  self.dist_coeffs)
                    pose_6dof = {'rvec': rvecs[0].tolist(), 'tvec': tvecs[0].tolist()}
                    marker_poses.append(pose_6dof)
                else:
                    marker_ids.append(count_num)
                    marker_poses.append(None)
                    count_num += 1
            else:
                marker_ids.append(count_num)
                marker_poses.append(None)
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

        mask_messages = [to_mask_msg(mask_in_roi, roi, w, h) for mask_in_roi,
                         roi in zip(masks_in_rois, rois)]

        masks_in_rois = reconstruct_masks(mask_messages)

        # Визуализация результата
        img_with_masks = draw_objects(img, scores=conf, objects_ids=class_ids, boxes=boxes, masks=masks_in_rois,
                                      draw_scores=True, draw_ids=True, draw_boxes=False, draw_masks=True, 
                                      palette=self.colors_palette, color_by_object_id=True )

      
        output_dir = "/home/angelika/Desktop/Cogmodel/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/output_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_image_path = os.path.join(output_dir, f"segmented_image_{self.request_counter}.png")
        cv2.imwrite(output_image_path, img_with_masks)

        boxes = [Box(x_min=int(box[0]), y_min=int(box[1]), x_max=int(box[2]), y_max=int(box[3])) for box in results.boxes.xyxy.cpu().numpy()]
        
        
        response = SegmentorResponse(
            num=num,
            scores=conf,
            classes_ids=class_ids,
            tracking_ids=marker_ids,
            boxes=boxes,
            marker_poses=marker_poses
        )

        return response



if __name__ == "__main__":
    segmentor = Segmentor()
    image_path = "/home/angelika/Desktop/Cogmodel/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/data/images/1727257198693604525.png"
    response = segmentor.segment_image(image_path)
    print(response)
