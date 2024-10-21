from pydantic.v1 import BaseModel
from langserve import RemoteRunnable
from langchain.output_parsers import PydanticOutputParser
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


class SegmentorResponse(BaseModel):
    num: int
    scores: List[float]
    classes_ids: List[int]
    tracking_ids: Optional[List[int]] = None 
    boxes: List[Box]
    # masks: List[Mask]
    analysis: dict
