import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2.aruco as aruco

#print(dir(aruco))
from masks import get_masks_in_rois, get_masks_rois, scale_image, reconstruct_masks
from visualization import draw_objects
from conversions import to_mask_msg
from seg_and_track import SegmentorResponse, Box, Pose
#from DoF_pose_estim_aruco import my_estimatePoseSingleMarkers, draw_pose_axis
# Classes
#names:
#  0 : 'box'
#  1 : 'container'
#  2 : 'person'
#  3 : 'shelf'


class Segmentor:
    def __init__(self):
        # Загружаем модель - переделать на hugginface
        self.model = YOLO("/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/weights/best_fisheye.pt")
        #print("Model loaded")
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
        self.aruco_params = aruco.DetectorParameters_create()
        self.camera_matrix = np.array([[580.77518, 0.0, 724.75002], [0.0, 580.77518, 570.98956], [0.0, 0.0, 1.0]])  
        self.dist_coeffs = np.array([0.927077, 0.141438, 0.000196, -8.7e-05, 0.001695, 1.257216, 
                                     0.354688, 0.015954])  
        
        # use Knew to scale the output for the change fx and fy
        # self.Knew = self.camera_matrix.copy()
        # self.Knew[(0,1), (0,1)] = 0.4 * self.Knew[(0,1), (0,1)]





    def segment_image(self, image_path):
        self.request_counter += 1

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        # Получаем результаты сегментации от модели
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

        marker_ids = []
        marker_poses = []
        count_num = 0
        marker_length = 0.08 # Length of the Aruco marker side
        
        #img_undistorted = cv2.fisheye.undistortImage(img, K = self.camera_matrix, D = self.dist_coeffs)


        for i, roi in enumerate(rois):
            x = int(roi[1].start)
            y = int(roi[0].start)
            #print(x, y)
            w_roi = int(roi[1].stop - roi[1].start)
            h_roi = int(roi[0].stop - roi[0].start)

            
     
            #roi_image = img_undistorted[y:y + h_roi, x:x + w_roi]
            roi_image = img[y:y + h_roi, x:x + w_roi]

            # Detect Aruco markers
            corners, ids, _ = aruco.detectMarkers(roi_image, self.aruco_dict)
            

            if ids is not None:
                aruco_mask = np.zeros(roi_image.shape[:2], dtype=np.uint8)
                pts = corners[0][0].astype(np.int32)
                #print(pts)
                cv2.fillPoly(aruco_mask, [pts], 1)

                if np.any(masks_in_rois[i] * aruco_mask):
                    marker_ids.append(ids[0][0])
                    corners_with_offset = [corner + [x, y] for corner in corners]


                    # Рассчет 6DoF позы
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners_with_offset, marker_length, self.camera_matrix, self.dist_coeffs)

                    
                    pose_6dof = {'rvec': rvecs[0].tolist(), 'tvec': tvecs[0].tolist()}
                    marker_poses.append(pose_6dof)

                    # Draw the Aruco marker corners
                    cv2.polylines(img, [pts + [x, y]], isClosed=True, color=(0, 255, 255), thickness=3)
                    cv2.putText(img, f"Aruco_id: {ids[0][0]}", (x + pts[0][0], y + pts[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # Draw 6DoF pose axes on the image
                    #img_with_pose = cv2.aruco.drawAxis(img_undistorted, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.08)
                    img_with_pose = cv2.aruco.drawAxis(img, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.08)
                                            
            else:
                marker_ids.append(count_num)
                pose_6dof = {'rvec': [[0, 0, 0]], 'tvec': [[0, 0, 0]]}  
                marker_poses.append(pose_6dof)
                count_num += 1
                print(count_num)

        # Удаление дубликатов marker_ids
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

        # Оставляем только классы 0 и 1
        filtered_indices = [i for i, class_id in enumerate(class_ids) if class_id in [0, 1]]
        #print(filtered_indices)
        #print(marker_ids)
        filtered_marker_ids = [marker_ids[i] for i in filtered_indices]
        
        filtered_marker_poses = [marker_poses[i] for i in filtered_indices]

        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_conf = [conf[i] for i in filtered_indices]
        filtered_class_ids = [class_ids[i] for i in filtered_indices]

        # check box_on_box
        box_on_box = False  

        for i, current_box in enumerate(filtered_boxes):
            if filtered_class_ids[i] == 0: 
                x_min, y_min, x_max, y_max = current_box
                top_region_y_max = y_min - 10  # Define the top area of the box

                
                for j, other_box in enumerate(filtered_boxes):
                    if i != j and filtered_class_ids[j] == 0:  
                        other_x_min, other_y_min, other_x_max, other_y_max = other_box
                        
                        
                        if other_x_min < x_max and other_x_max > x_min and other_y_max > top_region_y_max and other_y_min < y_min:
                            box_on_box = True
                            break  
                if box_on_box:
                    break 

      
        num = len(conf)
        if len(marker_ids) > num:
            marker_ids = marker_ids[:num]

        mask_messages = [to_mask_msg(mask_in_roi, roi, w, h) for mask_in_roi,
                        roi in zip(masks_in_rois, rois)]

        masks_in_rois = reconstruct_masks(mask_messages)
        output_dir = "/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/data/output"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #output_pose_image_path = os.path.join(output_dir, f"aruco_pose_image_{self.request_counter}.png")
        #cv2.imwrite(output_pose_image_path, img_undistorted)
        #cv2.imwrite(output_pose_image_path, img)

        # Визуализация результата
        img_with_masks = draw_objects(img, scores=conf, objects_ids=class_ids, boxes=boxes, masks=masks_in_rois,
                                    draw_scores=True, draw_ids=True, draw_boxes=False, draw_masks=True,
                                    palette=self.colors_palette, color_by_object_id=True)

        output_image_path = os.path.join(output_dir, f"segmented_image_{self.request_counter}.png")
        cv2.imwrite(output_image_path, img_with_masks)

        # Определяем, есть ли человек в кадре
        man_in_frame = 2 in class_ids

        # Создаем объекты Box и Pose
        boxes_response = [Box(x_min=int(box[0]), y_min=int(box[1]), x_max=int(box[2]), y_max=int(box[3])) for box in filtered_boxes]
        #print(filtered_marker_poses)

        poses_response = [
            Pose(rvec=pose['rvec'], tvec=pose['tvec']) if pose is not None else Pose(rvec=[0, 0, 0], tvec=[0, 0, 0])
            for pose in filtered_marker_poses
            ]
        
        print(poses_response)
        
        response = SegmentorResponse(
            num = len(filtered_conf),
            scores=filtered_conf,
            classes_ids=filtered_class_ids,
            tracking_ids=filtered_marker_ids,
            boxes = boxes_response,
            poses = poses_response,
            box_on_box = box_on_box,
            man_in_frame = man_in_frame
        )
        print(box_on_box)

        return response






if __name__ == "__main__":
    segmentor = Segmentor()
    image_path = "/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/data/images/1727257459670389985.png"
    response = segmentor.segment_image(image_path)
    print(response)
    
