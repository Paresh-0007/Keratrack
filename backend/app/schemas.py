from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    created_at: datetime
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class PredictionCreate(BaseModel):
    # file upload handled by FastAPI, so nothing here
    pass

class PredictionOut(BaseModel):
    id: int
    image_path: str
    predicted_stage: str
    confidence: float
    created_at: datetime
    class Config:
        orm_mode = True