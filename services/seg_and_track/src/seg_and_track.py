from pydantic.v1 import BaseModel
from typing import Optional, List


class Roi(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Mask(BaseModel):
    width: int
    height: int
    roi: Roi
    mask_in_roi: List[int]


class Box(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Pose(BaseModel):
    rvec: List[List[float]]  # Вектор поворота
    tvec: List[List[float]]  # Вектор трансляции

class BoxOutput(BaseModel):
    box_id: int
    placed_on_shelf_with_id: int


class Graph(BaseModel):
    id_1: Optional[int] = 0
    id_2: Optional[int] = 0
    rel_id: Optional[int] = 0
    class_name_1: Optional[str] = None
    rel_name: Optional[str] = None
    class_name_2: Optional[str] = None


class Shelf(BaseModel):
    shelf_id: int
    x: float
    y: float
    occupied_by_box_with_id: Optional[int] = -1
    pose: Pose


class SegmentorResponse(BaseModel):
    num: int
    scores: List[float]
    classes_ids: List[int]
    tracking_ids: Optional[List[int]] = None
    boxes: List[Box]
    poses: Optional[List[Pose]] = None  # Позиции объектов с ориентацией и трансляцией
    box_on_box: bool  
    man_in_frame: bool 
    box_container_on_floor: bool
    box_or_container_in_frame: bool
    right_size_flags: bool
    boxes_output: Optional[List[BoxOutput]]
    shelves: List[Shelf]
    graph_evr: Optional[Graph] = None






