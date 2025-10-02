from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, Date, Text, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    predictions = relationship("Prediction", back_populates="user")
    diet_assessments = relationship("DietAssessment", back_populates="user")
    diet_recommendations = relationship("DietRecommendation", back_populates="user")
    lifestyle_entries = relationship("LifestyleEntry", back_populates="user")
    food_logs = relationship("FoodLog", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String, nullable=False)
    predicted_stage = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")

# Diet Recommendation Models
class DietAssessment(Base):
    __tablename__ = "diet_assessments"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    age = Column(Integer)
    gender = Column(String(10))
    weight = Column(Float)  # in kg
    height = Column(Float)  # in cm
    activity_level = Column(String(20))  # sedentary, moderate, active
    dietary_restrictions = Column(JSON)  # allergies, preferences
    current_diet_pattern = Column(String(50))  # mediterranean, vegetarian, etc.
    health_conditions = Column(JSON)  # thyroid, pcos, etc.
    medications = Column(JSON)  # current medications
    assessment_date = Column(Date, default=datetime.utcnow)
    user = relationship("User", back_populates="diet_assessments")

class DietRecommendation(Base):
    __tablename__ = "diet_recommendations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    assessment_id = Column(Integer, ForeignKey("diet_assessments.id"))
    hair_stage_at_time = Column(String(20))  # hair stage when recommendation was made
    recommended_nutrients = Column(JSON)  # daily nutrient targets
    meal_plan = Column(JSON)  # weekly meal plan
    supplements = Column(JSON)  # recommended supplements
    foods_to_avoid = Column(JSON)  # foods to limit/avoid
    confidence_score = Column(Float)  # AI confidence in recommendation
    reasoning = Column(Text)  # explanation of recommendations
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    user = relationship("User", back_populates="diet_recommendations")

class LifestyleEntry(Base):
    __tablename__ = "lifestyle_entries"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    stress_level = Column(Integer)  # 1-10 scale
    sleep_hours = Column(Float)
    exercise_minutes = Column(Integer)
    water_intake = Column(Float)  # liters
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="lifestyle_entries")

class FoodLog(Base):
    __tablename__ = "food_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    date = Column(Date)
    meal_type = Column(String(20))  # breakfast, lunch, dinner, snack
    foods = Column(JSON)  # list of foods consumed
    estimated_nutrients = Column(JSON)  # calculated nutrient content
    logged_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="food_logs")