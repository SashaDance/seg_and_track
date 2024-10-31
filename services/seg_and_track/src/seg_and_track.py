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
    class_id: int
    box: List[int]
    pose: Optional[List[Pose]] = None 
    shelf_id: int

class ShelfOutput(BaseModel):
    shelf_id: int
    x: float
    y: float
    pose: Optional[List[Pose]] = None 


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
    boxes_output: List[BoxOutput]
    shelves: List[ShelfOutput]






