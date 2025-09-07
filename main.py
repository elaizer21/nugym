from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, schemas
from database import engine, SessionLocal
import os

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="FitSync Pro API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def root():
    return {"message": "FitSync Pro Backend is running!"}

# Example endpoint: Register user
@app.post("/register", response_model=schemas.UserResponse)
def register_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = models.get_password_hash(user.password)
    new_user = models.User(
        email=user.email,
        username=user.username,
        hashed_password=hashed_password,
        full_name=user.full_name,
        age=user.age,
        gender=user.gender,
        height=user.height,
        weight=user.weight,
        activity_level=user.activity_level,
        fitness_goal=user.fitness_goal,
        target_weight=user.target_weight,
        target_date=user.target_date
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# More endpoints (login, food log, exercise log, etc.) can be added similarly.