from sqlalchemy.orm import Session
from . import models, schemas
from typing import List, Optional

# User CRUD
def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate, hashed_password: str) -> models.User:
    db_user = models.User(
        name=user.name,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

# Prediction CRUD
def create_prediction(
    db: Session, 
    user_id: int, 
    image_path: str, 
    predicted_stage: str, 
    confidence: float
) -> models.Prediction:
    db_pred = models.Prediction(
        user_id=user_id,
        image_path=image_path,
        predicted_stage=predicted_stage,
        confidence=confidence
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)
    return db_pred

def get_user_predictions(db: Session, user_id: int) -> List[models.Prediction]:
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.user_id == user_id)
        .order_by(models.Prediction.created_at.desc())
        .all()
    )