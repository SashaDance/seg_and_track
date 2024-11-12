import os
import numpy as np
import cv2
import json
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2.aruco as aruco
from typing import List





#print(dir(aruco))
from masks import get_masks_in_rois, get_masks_rois, scale_image, reconstruct_masks
from visualization import draw_objects
from conversions import to_mask_msg
from seg_and_track import SegmentorResponse, Box, Pose, Graph
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
        #self.model = YOLO("/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/weights/best_fisheye.pt")
        model_path = hf_hub_download(repo_id="AnzhelikaK/yolov11_s_mini", filename="best.pt")
        self.model = YOLO(model_path)
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

        # Словарь эталонных размеров по id маркера (ширина x высота в метрах)
        self.aruco_size_reference = {
            313: (0.2, 0.2),  # box
            998: (0.34, 0.26),  # box
            999: (0.29, 0.19),  # container
            990: (0.38, 0.3)   # container
        }

        # Параметры Aruco
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self.aruco_params = aruco.DetectorParameters_create()
        self.camera_matrix = np.array([[580.77518, 0.0, 724.75002], [0.0, 580.77518, 570.98956], [0.0, 0.0, 1.0]])  
        self.dist_coeffs = np.array([0.927077, 0.141438, 0.000196, -8.7e-05, 0.001695, 1.257216, 
                                     0.354688, 0.015954])  
        self.dist_fish = np.array([0.927077, 0.141438, 0.000196, -8.7e-05]) 
        
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
                    img_with_pose = cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], 0.08)
                                            
            else:
                marker_ids.append(count_num)
                pose_6dof = {'rvec': [[0, 0, 0]], 'tvec': [[0, 0, 0]]}  
                marker_poses.append(pose_6dof)
                count_num += 1
                #print(count_num)

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

      

        
        # Оставляем только классы 0 и 1, а также добавляем полки для проверки
        filtered_indices = [i for i, class_id in enumerate(class_ids) if class_id in [0, 1]]
        shelves_indices = [i for i, class_id in enumerate(class_ids) if class_id == 3]

        # Отфильтрованные данные для коробок и контейнеров
        filtered_marker_ids = [marker_ids[i] for i in filtered_indices]
        filtered_marker_poses = [marker_poses[i] for i in filtered_indices]
        filtered_boxes = [boxes[i] for i in filtered_indices]
        filtered_conf = [conf[i] for i in filtered_indices]
        filtered_class_ids = [class_ids[i] for i in filtered_indices]


        # Данные для полок
        shelf_boxes = [boxes[i] for i in shelves_indices]

        # Проверка, находится ли коробка/контейнер на полу
        box_on_floor = False

        for i, current_box in enumerate(filtered_boxes):
            if filtered_class_ids[i] in [0, 1]:  # Проверяем только для коробок и контейнеров
                x_min, y_min, x_max, y_max = current_box
                below_region_y_min = y_min - 100  # Область ниже коробки/контейнера

                on_shelf = False
                for shelf_box in shelf_boxes:
                    shelf_x_min, shelf_y_min, shelf_x_max, shelf_y_max = shelf_box

                    # Проверка на наличие полки под коробкой/контейнером по критерию y
                    if shelf_y_max < y_min:  # только по y
                        on_shelf = True
                        break

                # Устанавливаем флаг на пол, если нет полки под текущей коробкой
                if not on_shelf:
                    #print("Параметры текущей коробки:", current_box)
                    #print("Параметры проверяемой полки:", shelf_box)
                    box_on_floor = True
                    break  # Достаточно одного флага для всей сцены
        

        undistorted_image, new_camera_matrix = undistort_image(img, self.camera_matrix, self.dist_fish)
    


 # --- BOX AND SHELF ASSOCIATION ---

        corners_shelf, ids_shelf, _ = aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params)
        #print(ids_shelf)
        marker_poses_shelf = []
        shelves = []  

    
        if ids_shelf is not None:
            rvecs_shelf, tvecs_shelf, _ = cv2.aruco.estimatePoseSingleMarkers(corners_shelf, marker_length, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids_shelf)):
                marker_id = ids_shelf[i][0]
                shelf_id = marker_id - 10 if 10 <= marker_id <= 17 else -1 #check aruco id
                #print(shelf_id)

                if shelf_id != -1:

                    pts_shelves = corners_shelf[i][0].astype(np.int32)  # Используем corners_shelf[i]
                    cv2.polylines(img, [pts_shelves], isClosed=True, color=(0, 255, 255), thickness=3)
                    cv2.putText(img, f"Aruco_id: {marker_id}", (pts_shelves[0][0], pts_shelves[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    pose_6dof_shelf = {'rvec': rvecs_shelf[i].tolist(), 'tvec': tvecs_shelf[i].tolist()}
                    marker_x, marker_y = tvecs_shelf[i][0][:2]
                    shelves.append({"shelf_id": shelf_id, "x": marker_x, "y": marker_y, "pose": pose_6dof_shelf, "occupied_by_box_with_id": -1})

                    # Получение координат маркера в пикселях
                    rvec, tvec = rvecs_shelf[i], tvecs_shelf[i]
                    imgpts, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), rvec, tvec, self.camera_matrix, self.dist_coeffs)
                    marker_x, marker_y = imgpts[0][0]


                    pose_6dof_shelf = {'rvec': rvecs_shelf[i].tolist(), 'tvec': tvecs_shelf[i].tolist()}
                    shelves.append({"shelf_id": shelf_id, "x": marker_x, "y": marker_y, "pose": pose_6dof_shelf, "occupied_by_box_with_id": -1})


        boxes_output = []
        aruco_to_shelf = {10 + i: i + 1 for i in range(6)}

        # Process Aruco markers and create shelf data
        for marker_id, marker_pose in zip(marker_ids, marker_poses):
            shelf_id = aruco_to_shelf.get(marker_id, -1)
            if shelf_id != -1:
                marker_x, marker_y = marker_pose['tvec'][0][:2]
                shelves.append({"shelf_id": shelf_id, "x": marker_x, "y": marker_y, "occupied_by_box_with_id": -1})  # Store marker position


        # Process boxes and associate them with shelves based on Aruco marker proximity
        for i, (box_id, class_id, box) in enumerate(zip(filtered_marker_ids, filtered_class_ids, filtered_boxes)):
            if class_id in [0, 1]:  # Box or container
                box_x_min, box_y_min, box_x_max, box_y_max = box
                box_center_x = (box_x_min + box_x_max) / 2
                placed_on_shelf = -1

                for j, shelf_data in enumerate(shelves):
                    shelf_id = shelf_data['shelf_id']
                    marker_x = shelf_data['x']
                    marker_y = shelf_data['y']
                    vertical_range = 200  # Adjust as needed
                    horizontal_range = 100 # Adjust as needed

                    # Calculate search area coordinates
                    search_x_min = int(marker_x - horizontal_range)
                    search_x_max = int(marker_x + horizontal_range)
                    search_y_min = int(marker_y - vertical_range)  # Start above the marker
                    search_y_max = int(marker_y) # End at the marker


                    # Draw the search area
                    cv2.rectangle(img, (search_x_min, search_y_min), (search_x_max, search_y_max), (0, 255, 0), 2)


                    if (abs(box_center_x - marker_x) < horizontal_range and  # Horizontal proximity check
                            marker_y - vertical_range <= box_y_max <= marker_y): # Vertical proximity (above the marker)
                        placed_on_shelf = shelf_id
                        shelves[j]["occupied_by_box_with_id"] = box_id
                        break

                boxes_output.append({"box_id": box_id, "placed_on_shelf_with_id": placed_on_shelf})

        #print(f"boxes_output {boxes_output}")

        #print(f"shelves {shelves}")



        # Проверка размеров для каждого бокса
             
        right_size_flags = True
        for marker_id, box in zip(filtered_marker_ids, filtered_boxes):
            x_min, y_min, x_max, y_max = box
            
            # Вычисляем ширину и высоту бокса в метрах на выпрямленном изображении
            detected_width = (x_max - x_min) / new_camera_matrix[0, 0]
            detected_height = (y_max - y_min) / new_camera_matrix[1, 1]

            # Проверяем, если id маркера есть в эталонных размерах
            if marker_id in self.aruco_size_reference:
                ref_width, ref_height = self.aruco_size_reference[marker_id]
                # Допустимый диапазон для размера: отклонение в 10%
                tolerance = 0.3
                is_correct_size = (
                    (1 - tolerance) * ref_width <= detected_width <= (1 + tolerance) * ref_width and
                    (1 - tolerance) * ref_height <= detected_height <= (1 + tolerance) * ref_height
                )

                print(
                    f'detected w: {detected_width}, actual: {ref_width}, x_max - x_min: {x_max - x_min}',
                    f'detected h: {detected_height}, actual: {ref_height}, y_max - y_min: {y_max - y_min}',
                    sep='\n'
                )
                if not is_correct_size:
                    # Если хотя бы одно значение неверно, сразу возвращаем False
                    right_size_flags = False
            else:
                # Если id маркера нет в эталонных размерах, возвращаем False
                right_size_flags = False

        relationships: List[Graph] = []

        # ______ BOX_ON__BOX
        box_on_box = False
        message = None

        for i, current_box in enumerate(filtered_boxes):
            if filtered_class_ids[i] == 0:  # проверка на коробку
                x_min, y_min, x_max, y_max = current_box
                top_region_y_max = y_min - 10  # Определение верхней области коробки

                for j, other_box in enumerate(filtered_boxes):
                    if i != j and filtered_class_ids[j] == 0:  # Проверка на вторую коробку
                        other_x_min, other_y_min, other_x_max, other_y_max = other_box

                        # Проверка нахождения одной коробки на другой
                        if other_x_min < x_max and other_x_max > x_min and other_y_max > top_region_y_max and other_y_min < y_min:
                            box_on_box = True

                            # Создаем объект Graph для связи и добавляем его в список
                            message = Graph(
                                id_1=filtered_marker_ids[i],  # ID первой коробки
                                id_2=filtered_marker_ids[j],  # ID второй коробки
                                rel_id=2,
                                class_name_1="box",
                                rel_name="on_top",
                                class_name_2="box",
                            )
                            relationships.append(message)

                            # Переход к следующей коробке, если связь найдена
                            break  
                if box_on_box:
                    break  
        #print(message)
           



      
        num = len(conf)
        if len(marker_ids) > num:
            marker_ids = marker_ids[:num]

        mask_messages = [to_mask_msg(mask_in_roi, roi, w, h) for mask_in_roi,
                        roi in zip(masks_in_rois, rois)]

        masks_in_rois = reconstruct_masks(mask_messages)
        output_dir = "D:/pythonProject/seg_and_track/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/data/output"

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

        # Определение, присутствует ли коробка или контейнер в кадре
        box_or_container_in_frame = any(cls_id in [0, 1] for cls_id in class_ids)
        #print(box_on_floor)

        # Создаем объекты Box и Pose
        boxes_response = [Box(x_min=int(box[0]), y_min=int(box[1]), x_max=int(box[2]), y_max=int(box[3])) for box in filtered_boxes]
        poses_response = [
            Pose(rvec=pose['rvec'], tvec=pose['tvec']) if pose is not None else Pose(rvec=[0, 0, 0], tvec=[0, 0, 0])
            for pose in filtered_marker_poses
        ]

        

        # Формируем response с новыми флагами
        response = SegmentorResponse(
            num=len(filtered_conf),
            scores=filtered_conf,
            classes_ids=filtered_class_ids,
            tracking_ids=filtered_marker_ids,
            boxes=boxes_response,
            poses=poses_response,
            box_on_box=box_on_box,
            man_in_frame=man_in_frame,
            box_container_on_floor=box_on_floor,
            box_or_container_in_frame=box_or_container_in_frame,
            right_size_flags = right_size_flags,
            boxes_output = boxes_output,
            shelves = shelves,
            graph_evr = message 
        )





        print(response)
        json_output_path = os.path.join(
            "D:/pythonProject/seg_and_track/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/data/output",
            f"segmentation_result_{self.request_counter}.json"
        )
        with open(json_output_path, 'w') as json_file:
            json.dump(response.dict(), json_file, ensure_ascii=False, indent=4)

        return response
    

def undistort_image(image, camera_matrix, distortion_coefficients):
    # Исправляем искажения на изображении
    h, w = image.shape[:2]
    new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera_matrix, distortion_coefficients, (w, h), None
    )
    undistorted_image = cv2.fisheye.undistortImage(
        image, camera_matrix, distortion_coefficients, Knew=new_camera_matrix
    )
    return undistorted_image, new_camera_matrix





if __name__ == "__main__":
    segmentor = Segmentor()
    # image_path = "/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/data/images/wp2.png"
    #image_path = "/home/angelika/Desktop/7_term/feat-seg_and_track/services/seg_and_track/tests/data/images/1727257199413594933.png"
    image_path = 'D:/pythonProject/seg_and_track/R-D-AC_robotic_integration-feat-seg_and_track/services/seg_and_track/tests/data/images/wp2.png'
    response = segmentor.segment_image(image_path)
    # print(response)
    
