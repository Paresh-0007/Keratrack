import os
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from PIL import Image
from . import models, schemas, crud, auth, database, ml_interface as ml_inference
from fastapi.staticfiles import StaticFiles


# ====== Setup ======
app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

models.Base.metadata.create_all(bind=database.engine)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ====== Auth ======
@app.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = auth.get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = auth.get_password_hash(user.password)
    new_user = models.User(
        name=user.name, email=user.email, hashed_password=hashed
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = auth.create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from jose import JWTError, jwt
    from .auth import SECRET_KEY, ALGORITHM
    credentials_exception = HTTPException(
        status_code=401, detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = auth.get_user_by_email(db, email)
    if user is None:
        raise credentials_exception
    return user

# ====== Prediction ======
@app.post("/predict", response_model=schemas.PredictionOut)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Save image
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    image = Image.open(file_path).convert("RGB")
    # Run model
    stage, confidence = ml_inference.predict(image)
    # Store in DB
    pred = models.Prediction(
        user_id=current_user.id,
        image_path=file_path,
        predicted_stage=stage,
        confidence=confidence
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    return pred

# ====== History ======
@app.get("/history", response_model=list[schemas.PredictionOut])
def get_history(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    return db.query(models.Prediction).filter(models.Prediction.user_id == current_user.id).order_by(models.Prediction.created_at.desc()).all()

# ====== Diet Recommendations ======
from . import diet_ai

diet_engine = diet_ai.KeratrackDietAI()

@app.post("/diet/assessment", response_model=schemas.DietAssessmentOut)
def create_diet_assessment(
    assessment: schemas.DietAssessmentCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Create a new diet assessment for the user"""
    db_assessment = models.DietAssessment(
        user_id=current_user.id,
        age=assessment.age,
        gender=assessment.gender,
        weight=assessment.weight,
        height=assessment.height,
        activity_level=assessment.activity_level,
        dietary_restrictions=assessment.dietary_restrictions,
        current_diet_pattern=assessment.current_diet_pattern,
        health_conditions=assessment.health_conditions,
        medications=assessment.medications
    )
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    return db_assessment

@app.get("/diet/recommendations", response_model=schemas.DietRecommendationOut)
def get_diet_recommendations(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get AI-powered diet recommendations for the user"""
    
    # Get the latest assessment
    latest_assessment = db.query(models.DietAssessment).filter(
        models.DietAssessment.user_id == current_user.id
    ).order_by(models.DietAssessment.assessment_date.desc()).first()
    
    if not latest_assessment:
        raise HTTPException(
            status_code=400, 
            detail="Please complete a diet assessment first"
        )
    
    # Check if we have recent recommendations
    recent_recommendation = db.query(models.DietRecommendation).filter(
        models.DietRecommendation.user_id == current_user.id,
        models.DietRecommendation.is_active == True
    ).order_by(models.DietRecommendation.created_at.desc()).first()
    
    # Generate new recommendations if none exist or they're old (>30 days)
    if (not recent_recommendation or 
        (datetime.utcnow() - recent_recommendation.created_at).days > 30):
        
        # Prepare assessment data for AI
        assessment_data = {
            'age': latest_assessment.age,
            'gender': latest_assessment.gender,
            'weight': latest_assessment.weight,
            'height': latest_assessment.height,
            'activity_level': latest_assessment.activity_level,
            'dietary_restrictions': latest_assessment.dietary_restrictions,
            'current_diet_pattern': latest_assessment.current_diet_pattern,
            'health_conditions': latest_assessment.health_conditions,
            'medications': latest_assessment.medications
        }
        
        # Generate AI recommendations
        ai_recommendations = diet_engine.generate_complete_recommendation(
            db, current_user.id, assessment_data
        )
        
        # Get current hair stage
        latest_prediction = db.query(models.Prediction).filter(
            models.Prediction.user_id == current_user.id
        ).order_by(models.Prediction.created_at.desc()).first()
        
        current_stage = latest_prediction.predicted_stage if latest_prediction else "LEVEL_4"
        
        # Deactivate old recommendations
        db.query(models.DietRecommendation).filter(
            models.DietRecommendation.user_id == current_user.id
        ).update({"is_active": False})
        
        # Save new recommendation
        db_recommendation = models.DietRecommendation(
            user_id=current_user.id,
            assessment_id=latest_assessment.id,
            hair_stage_at_time=current_stage,
            recommended_nutrients=ai_recommendations['nutrients'],
            meal_plan=ai_recommendations['meal_plan'],
            supplements=ai_recommendations['supplements'],
            foods_to_avoid=[],  # You can expand this
            confidence_score=ai_recommendations['confidence_score'],
            reasoning=ai_recommendations['reasoning']
        )
        
        db.add(db_recommendation)
        db.commit()
        db.refresh(db_recommendation)
        
        return db_recommendation
    
    return recent_recommendation

@app.post("/diet/lifestyle", response_model=schemas.LifestyleEntryOut)
def log_lifestyle_entry(
    entry: schemas.LifestyleEntryCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Log daily lifestyle factors"""
    db_entry = models.LifestyleEntry(
        user_id=current_user.id,
        date=entry.date,
        stress_level=entry.stress_level,
        sleep_hours=entry.sleep_hours,
        exercise_minutes=entry.exercise_minutes,
        water_intake=entry.water_intake,
        notes=entry.notes
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

@app.get("/diet/lifestyle/history", response_model=list[schemas.LifestyleEntryOut])
def get_lifestyle_history(
    days: int = 30,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get lifestyle entry history"""
    from datetime import date, timedelta
    start_date = date.today() - timedelta(days=days)
    
    return db.query(models.LifestyleEntry).filter(
        models.LifestyleEntry.user_id == current_user.id,
        models.LifestyleEntry.date >= start_date
    ).order_by(models.LifestyleEntry.date.desc()).all()

@app.post("/diet/food-log", response_model=schemas.FoodLogOut)
def log_food_entry(
    food_log: schemas.FoodLogCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Log daily food intake"""
    db_food_log = models.FoodLog(
        user_id=current_user.id,
        date=food_log.date,
        meal_type=food_log.meal_type,
        foods=food_log.foods,
        estimated_nutrients=food_log.estimated_nutrients
    )
    db.add(db_food_log)
    db.commit()
    db.refresh(db_food_log)
    return db_food_log

@app.get("/diet/progress-analysis")
def get_diet_progress_analysis(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Get analysis of diet effectiveness on hair health"""
    
    # Get hair progression analysis
    hair_analysis = diet_engine.analyze_user_hair_progression(db, current_user.id)
    
    # Get lifestyle data for correlation
    lifestyle_entries = db.query(models.LifestyleEntry).filter(
        models.LifestyleEntry.user_id == current_user.id
    ).order_by(models.LifestyleEntry.date.desc()).limit(30).all()
    
    # Simple correlation analysis
    correlations = {}
    if lifestyle_entries:
        stress_levels = [entry.stress_level for entry in lifestyle_entries]
        sleep_hours = [entry.sleep_hours for entry in lifestyle_entries]
        
        correlations = {
            "avg_stress_level": sum(stress_levels) / len(stress_levels),
            "avg_sleep_hours": sum(sleep_hours) / len(sleep_hours),
            "lifestyle_consistency": len(lifestyle_entries)  # More entries = more consistent tracking
        }
    
    return {
        "hair_analysis": hair_analysis,
        "lifestyle_correlations": correlations,
        "recommendations_status": "active" if hair_analysis.get('trend') == 'improving' else "needs_adjustment"
    }