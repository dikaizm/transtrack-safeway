# Import all pipelines here so their @register_pipeline decorators fire
from pipelines.yolo import YOLOPipeline
from pipelines.rtdetr import RTDETRPipeline
from pipelines.rfdetr import RFDETRPipeline

__all__ = ["YOLOPipeline", "RTDETRPipeline", "RFDETRPipeline"]
