import os
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