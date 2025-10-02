from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, date

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

# Diet Assessment Schemas
class DietAssessmentCreate(BaseModel):
    age: int
    gender: str  # "male" or "female"
    weight: float  # kg
    height: float  # cm
    activity_level: str  # "sedentary", "moderate", "active"
    dietary_restrictions: List[str] = []  # ["vegetarian", "gluten_free", "dairy_free", etc.]
    current_diet_pattern: Optional[str] = None  # "mediterranean", "vegetarian", "standard"
    health_conditions: List[str] = []  # ["thyroid", "pcos", "diabetes", etc.]
    medications: List[str] = []  # current medications

class DietAssessmentOut(BaseModel):
    id: int
    user_id: int
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str
    dietary_restrictions: List[str]
    current_diet_pattern: Optional[str]
    health_conditions: List[str]
    medications: List[str]
    assessment_date: date
    class Config:
        orm_mode = True

# Diet Recommendation Schemas
class DietRecommendationOut(BaseModel):
    id: int
    user_id: int
    hair_stage_at_time: str
    recommended_nutrients: Dict[str, Any]
    meal_plan: Dict[str, Any]
    supplements: List[Dict[str, Any]]
    foods_to_avoid: List[str]
    confidence_score: float
    reasoning: str
    created_at: datetime
    is_active: bool
    class Config:
        orm_mode = True

# Lifestyle Entry Schemas
class LifestyleEntryCreate(BaseModel):
    date: date
    stress_level: int  # 1-10
    sleep_hours: float
    exercise_minutes: int
    water_intake: float  # liters
    notes: Optional[str] = None

class LifestyleEntryOut(BaseModel):
    id: int
    user_id: int
    date: date
    stress_level: int
    sleep_hours: float
    exercise_minutes: int
    water_intake: float
    notes: Optional[str]
    created_at: datetime
    class Config:
        orm_mode = True

# Food Log Schemas
class FoodLogCreate(BaseModel):
    date: date
    meal_type: str  # "breakfast", "lunch", "dinner", "snack"
    foods: List[str]  # list of food items
    estimated_nutrients: Optional[Dict[str, float]] = None

class FoodLogOut(BaseModel):
    id: int
    user_id: int
    date: date
    meal_type: str
    foods: List[str]
    estimated_nutrients: Optional[Dict[str, float]]
    logged_at: datetime
    class Config:
        orm_mode = True