from pydantic import BaseModel

class Evaluation(BaseModel):
    fitness: float
    additional_data: dict[str, str] = {}