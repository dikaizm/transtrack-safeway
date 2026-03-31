# Import all pipelines here so their @register_pipeline decorators fire
from pipelines.yolo import YOLOPipeline
from pipelines.rtdetr import RTDETRPipeline
from pipelines.segformer import SegFormerPipeline

__all__ = ["YOLOPipeline", "RTDETRPipeline", "SegFormerPipeline"]
