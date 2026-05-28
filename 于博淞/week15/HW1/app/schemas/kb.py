from pydantic import BaseModel


class KBCreate(BaseModel):
    name: str


class KBResponse(BaseModel):
    id: int
    name: str

    model_config = {"from_attributes": True}
