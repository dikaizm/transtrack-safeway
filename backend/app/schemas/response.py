from typing import Any, Literal

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    status: Literal["success"] = "success"
    message: str
    data: Any | None = None

    model_config = {"json_encoders": {}, "populate_by_name": True}


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str
