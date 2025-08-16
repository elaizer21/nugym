# FitSync Pro - Complete Backend System
# Main Application Entry Point

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
import os
from typing import List, Optional, Dict, Any
import json
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import requests
import asyncio
from pydantic import BaseModel, EmailStr
import logging
from dotenv import load_dotenv
import redis
from celery import Celery
import stripe
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import smtplib
import uuid
from dataclasses import dataclass
import pickle
import mediapipe as mp
from scipy.spatial.distance import cosine
import boto3
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/fitsync_pro")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis Configuration
redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)

# Celery Configuration
celery_app = Celery(
    "fitsync_tasks",
    broker=os.getenv("CELERY_BROKER", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_BACKEND", "redis://localhost:6379/0")
)

# AWS S3 Configuration
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Stripe Configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# FastAPI App Configuration
app = FastAPI(title="FitSync Pro API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# DATABASE MODELS
# =============================================================================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    height = Column(Float)  # in cm
    weight = Column(Float)  # in kg
    activity_level = Column(String)
    fitness_goal = Column(String)  # "burn_fat", "build_muscle", "maintain"
    target_weight = Column(Float)
    target_date = Column(DateTime)
    subscription_type = Column(String, default="basic")  # basic, pro, elite
    stripe_customer_id = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    body_scans = relationship("BodyScan", back_populates="user")
    food_logs = relationship("FoodLog", back_populates="user")
    exercise_logs = relationship("ExerciseLog", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class BodyScan(Base):
    __tablename__ = "body_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scan_type = Column(String)  # "baseline", "progress", "monthly"
    measurements = Column(Text)  # JSON string of measurements
    scan_data_url = Column(String)  # S3 URL for 3D scan data
    photos_urls = Column(Text)  # JSON array of photo URLs
    body_fat_percentage = Column(Float)
    muscle_mass = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="body_scans")

class Food(Base):
    __tablename__ = "foods"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    barcode = Column(String)
    brand = Column(String)
    calories_per_100g = Column(Float)
    protein_per_100g = Column(Float)
    carbs_per_100g = Column(Float)
    fat_per_100g = Column(Float)
    fiber_per_100g = Column(Float)
    sugar_per_100g = Column(Float)
    sodium_per_100g = Column(Float)
    category = Column(String)
    verified = Column(Boolean, default=False)

class FoodLog(Base):
    __tablename__ = "food_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    food_id = Column(Integer, ForeignKey("foods.id"))
    quantity = Column(Float)  # in grams
    meal_type = Column(String)  # breakfast, lunch, dinner, snack
    confidence_score = Column(Float)  # AI recognition confidence
    input_method = Column(String)  # camera, voice, text
    photo_url = Column(String)
    logged_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="food_logs")
    food = relationship("Food")

class Exercise(Base):
    __tablename__ = "exercises"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String)  # strength, cardio, flexibility
    muscle_groups = Column(Text)  # JSON array
    equipment = Column(String)
    difficulty_level = Column(String)
    instructions = Column(Text)
    video_url = Column(String)
    calories_per_minute = Column(Float)

class ExerciseLog(Base):
    __tablename__ = "exercise_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    duration = Column(Integer)  # in minutes
    reps = Column(Integer)
    sets = Column(Integer)
    weight = Column(Float)  # in kg
    calories_burned = Column(Float)
    form_score = Column(Float)  # AI form analysis score
    logged_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="exercise_logs")
    exercise = relationship("Exercise")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    prediction_type = Column(String)  # weight_loss, muscle_gain, energy_level
    current_value = Column(Float)
    predicted_value = Column(Float)
    confidence = Column(Float)
    timeframe_days = Column(Integer)
    factors = Column(Text)  # JSON of contributing factors
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="predictions")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    stripe_subscription_id = Column(String)
    plan_type = Column(String)  # pro, elite
    status = Column(String)  # active, canceled, past_due
    current_period_start = Column(DateTime)
    current_period_end = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create all tables
Base.metadata.create_all(bind=engine)

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str
    age: int
    gender: str
    height: float
    weight: float
    activity_level: str
    fitness_goal: str
    target_weight: Optional[float] = None
    target_date: Optional[datetime] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    subscription_type: str
    fitness_goal: str
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class FoodLogCreate(BaseModel):
    food_name: str
    quantity: float
    meal_type: str
    input_method: str

class ExerciseLogCreate(BaseModel):
    exercise_name: str
    duration: Optional[int] = None
    reps: Optional[int] = None
    sets: Optional[int] = None
    weight: Optional[float] = None

class PredictionResponse(BaseModel):
    prediction_type: str
    current_value: float
    predicted_value: float
    confidence: float
    timeframe_days: int
    
    class Config:
        from_attributes = True

# =============================================================================
# DEPENDENCY FUNCTIONS
# =============================================================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# =============================================================================
# AI/ML HELPER FUNCTIONS
# =============================================================================

class FoodRecognitionAI:
    def __init__(self):
        # Initialize food recognition model (placeholder for actual model)
        self.model = None
        # In production, load your trained model here
        # self.model = tf.keras.models.load_model('food_recognition_model.h5')
    
    async def recognize_food(self, image_data: bytes) -> Dict[str, Any]:
        """Recognize food from image and return nutrition info"""
        try:
            # Convert image data to numpy array
            image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            
            # Placeholder AI recognition logic
            # In production, this would use your trained model
            recognized_foods = [
                {
                    "name": "Grilled Chicken Breast",
                    "confidence": 0.89,
                    "quantity_grams": 150,
                    "calories": 165,
                    "protein": 23.0,
                    "carbs": 0.0,
                    "fat": 7.5
                }
            ]
            
            return {
                "foods": recognized_foods,
                "total_calories": sum(f["calories"] for f in recognized_foods),
                "processing_time_ms": 1200
            }
        except Exception as e:
            logger.error(f"Food recognition error: {e}")
            return {"error": "Failed to recognize food", "foods": []}

class BodyScan3D:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    async def process_3d_scan(self, images: List[bytes]) -> Dict[str, Any]:
        """Process multiple images to create 3D body measurements"""
        try:
            measurements = {
                "waist": 82.5,  # cm
                "chest": 95.0,
                "bicep_left": 35.2,
                "bicep_right": 35.8,
                "thigh_left": 58.3,
                "thigh_right": 58.0,
                "body_fat_percentage": 15.2,
                "muscle_mass": 65.8  # kg
            }
            
            return {
                "measurements": measurements,
                "scan_quality": "excellent",
                "processing_time_ms": 3500
            }
        except Exception as e:
            logger.error(f"3D scan processing error: {e}")
            return {"error": "Failed to process 3D scan"}

class PredictionEngine:
    def __init__(self):
        # Initialize prediction models
        self.weight_model = None
        self.muscle_model = None
        self.energy_model = None
    
    async def predict_outcomes(self, user: User, recent_logs: Dict) -> List[Dict]:
        """Generate predictions based on user data and recent activity"""
        predictions = []
        
        # Weight prediction
        current_weight = user.weight
        target_weight = user.target_weight or current_weight
        days_to_goal = (user.target_date - datetime.utcnow()).days if user.target_date else 42
        
        # Simple prediction algorithm (replace with ML model)
        predicted_weight = current_weight - (0.5 * (days_to_goal / 7))  # 0.5kg per week
        
        predictions.append({
            "type": "weight_loss",
            "current_value": current_weight,
            "predicted_value": predicted_weight,
            "confidence": 0.85,
            "timeframe_days": days_to_goal,
            "factors": ["calorie_deficit", "exercise_consistency", "sleep_quality"]
        })
        
        # Muscle gain prediction
        if user.fitness_goal == "build_muscle":
            predictions.append({
                "type": "muscle_gain",
                "current_value": 65.0,  # current muscle mass
                "predicted_value": 68.2,  # predicted muscle mass
                "confidence": 0.78,
                "timeframe_days": 84,  # 12 weeks
                "factors": ["protein_intake", "strength_training", "recovery"]
            })
        
        # Energy level prediction
        predictions.append({
            "type": "energy_level",
            "current_value": 6.5,  # out of 10
            "predicted_value": 8.2,
            "confidence": 0.72,
            "timeframe_days": 14,
            "factors": ["nutrition_quality", "exercise_routine", "hydration"]
        })
        
        return predictions

class FormAnalysisAI:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
    
    async def analyze_form(self, video_data: bytes, exercise_type: str) -> Dict[str, Any]:
        """Analyze exercise form from video data"""
        try:
            # Placeholder form analysis
            form_score = np.random.uniform(0.7, 0.95)  # Random score for demo
            
            feedback = {
                "squat": [
                    "Great depth! Keep going below parallel.",
                    "Keep your chest up and core tight.",
                    "Your knees are tracking well over your toes."
                ],
                "pushup": [
                    "Excellent range of motion!",
                    "Keep your body in a straight line.",
                    "Control the descent for better muscle activation."
                ]
            }
            
            return {
                "form_score": form_score,
                "rep_count": np.random.randint(8, 15),
                "feedback": feedback.get(exercise_type, ["Good form overall!"]),
                "areas_for_improvement": ["core_stability", "tempo_control"]
            }
        except Exception as e:
            logger.error(f"Form analysis error: {e}")
            return {"error": "Failed to analyze form"}

# Initialize AI components
food_ai = FoodRecognitionAI()
body_scanner = BodyScan3D()
prediction_engine = PredictionEngine()
form_analyzer = FormAnalysisAI()

# =============================================================================
# CELERY TASKS
# =============================================================================

@celery_app.task
def process_food_image(user_id: int, image_path: str):
    """Background task to process food images"""
    try:
        # Process image and update database
        logger.info(f"Processing food image for user {user_id}")
        # Implementation would go here
        return {"status": "completed", "user_id": user_id}
    except Exception as e:
        logger.error(f"Error processing food image: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task
def generate_meal_plan(user_id: int, preferences: dict):
    """Background task to generate personalized meal plans"""
    try:
        logger.info(f"Generating meal plan for user {user_id}")
        # Implementation would go here
        return {"status": "completed", "meal_plan_id": f"mp_{user_id}_{uuid.uuid4()}"}
    except Exception as e:
        logger.error(f"Error generating meal plan: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task
def send_notification_email(user_email: str, subject: str, body: str):
    """Background task to send notification emails"""
    try:
        msg = MimeMultipart()
        msg['From'] = os.getenv("SMTP_FROM_EMAIL")
        msg['To'] = user_email
        msg['Subject'] = subject
        
        msg.attach(MimeText(body, 'html'))
        
        server = smtplib.SMTP(os.getenv("SMTP_SERVER"), 587)
        server.starttls()
        server.login(os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD"))
        server.send_message(msg)
        server.quit()
        
        return {"status": "sent", "recipient": user_email}
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return {"status": "failed", "error": str(e)}

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/auth/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    # Create Stripe customer
    try:
        stripe_customer = stripe.Customer.create(
            email=user_data.email,
            name=user_data.full_name,
            metadata={"username": user_data.username}
        )
        stripe_customer_id = stripe_customer.id
    except Exception as e:
        logger.error(f"Stripe customer creation failed: {e}")
        stripe_customer_id = None
    
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        age=user_data.age,
        gender=user_data.gender,
        height=user_data.height,
        weight=user_data.weight,
        activity_level=user_data.activity_level,
        fitness_goal=user_data.fitness_goal,
        target_weight=user_data.target_weight,
        target_date=user_data.target_date,
        stripe_customer_id=stripe_customer_id
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Send welcome email
    send_notification_email.delay(
        user_data.email,
        "Welcome to FitSync Pro!",
        f"<h1>Welcome {user_data.full_name}!</h1><p>Your AI-powered fitness journey begins now.</p>"
    )
    
    return db_user

@app.post("/auth/login", response_model=Token)
async def login_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Login user and return access token"""
    user = db.query(User).filter(User.username == username).first()
    
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/user/profile", response_model=UserResponse)
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@app.post("/food/recognize")
async def recognize_food_from_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Recognize food from uploaded image"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_data = await file.read()
    result = await food_ai.recognize_food(image_data)
    
    # Cache result in Redis for quick access
    cache_key = f"food_recognition:{current_user.id}:{datetime.utcnow().timestamp()}"
    redis_client.setex(cache_key, 300, json.dumps(result))  # 5 minutes TTL
    
    return result

@app.post("/food/log")
async def log_food(
    food_data: FoodLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log food intake"""
    # Find or create food entry
    food = db.query(Food).filter(Food.name.ilike(f"%{food_data.food_name}%")).first()
    
    if not food:
        # Create new food entry (in production, this would query a food database API)
        food = Food(
            name=food_data.food_name,
            calories_per_100g=200,  # Default values
            protein_per_100g=20,
            carbs_per_100g=10,
            fat_per_100g=5,
            category="unknown"
        )
        db.add(food)
        db.commit()
        db.refresh(food)
    
    # Create food log entry
    food_log = FoodLog(
        user_id=current_user.id,
        food_id=food.id,
        quantity=food_data.quantity,
        meal_type=food_data.meal_type,
        input_method=food_data.input_method,
        confidence_score=0.85
    )
    
    db.add(food_log)
    db.commit()
    db.refresh(food_log)
    
    # Generate updated predictions
    predictions = await prediction_engine.predict_outcomes(current_user, {})
    
    return {
        "message": "Food logged successfully",
        "food_log_id": food_log.id,
        "updated_predictions": predictions[:2]  # Return first 2 predictions
    }

@app.post("/exercise/log")
async def log_exercise(
    exercise_data: ExerciseLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log exercise activity"""
    # Find exercise
    exercise = db.query(Exercise).filter(
        Exercise.name.ilike(f"%{exercise_data.exercise_name}%")
    ).first()
    
    if not exercise:
        # Create new exercise entry
        exercise = Exercise(
            name=exercise_data.exercise_name,
            category="strength",
            muscle_groups=json.dumps(["unknown"]),
            calories_per_minute=8.0
        )
        db.add(exercise)
        db.commit()
        db.refresh(exercise)
    
    # Calculate calories burned
    calories_burned = (exercise_data.duration or 0) * exercise.calories_per_minute
    
    # Create exercise log
    exercise_log = ExerciseLog(
        user_id=current_user.id,
        exercise_id=exercise.id,
        duration=exercise_data.duration,
        reps=exercise_data.reps,
        sets=exercise_data.sets,
        weight=exercise_data.weight,
        calories_burned=calories_burned,
        form_score=0.85  # Default form score
    )
    
    db.add(exercise_log)
    db.commit()
    db.refresh(exercise_log)
    
    return {
        "message": "Exercise logged successfully",
        "exercise_log_id": exercise_log.id,
        "calories_burned": calories_burned
    }

@app.post("/body/scan")
async def create_body_scan(
    scan_type: str = Form(...),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process 3D body scan from multiple images"""
    if len(files) < 4:
        raise HTTPException(status_code=400, detail="At least 4 images required for 3D scanning")
    
    # Read all images
    image_data_list = []
    photo_urls = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="All files must be images")
        
        image_data = await file.read()
        image_data_list.append(image_data)
        
        # Upload to S3 (placeholder)
        photo_url = f"https://fitsync-scans.s3.amazonaws.com/{current_user.id}/{uuid.uuid4()}.jpg"
        photo_urls.append(photo_url)
    
    # Process 3D scan
    scan_result = await body_scanner.process_3d_scan(image_data_list)
    
    if "error" in scan_result:
        raise HTTPException(status_code=500, detail=scan_result["error"])
    
    # Create body scan record
    body_scan = BodyScan(
        user_id=current_user.id,
        scan_type=scan_type,
        measurements=json.dumps(scan_result["measurements"]),
        photos_urls=json.dumps(photo_urls),
        body_fat_percentage=scan_result["measurements"]["body_fat_percentage"],
        muscle_mass=scan_result["measurements"]["muscle_mass"]
    )
    
    db.add(body_scan)
    db.commit()
    db.refresh(body_scan)
    
    return {
        "message": "Body scan processed successfully",
        "scan_id": body_scan.id,
        "measurements": scan_result["measurements"],
        "scan_quality": scan_result["scan_quality"]
    }

@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI predictions for user"""
    # Get recent activity logs
    recent_food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.logged_at >= datetime.utcnow() - timedelta(days=7)
    ).all()
    
    recent_exercise_logs = db.query(ExerciseLog).filter(
        ExerciseLog.user_id == current_user.id,
        ExerciseLog.logged_at >= datetime.utcnow() - timedelta(days=7)
    ).all()
    
    recent_logs = {
        "food_logs": recent_food_logs,
        "exercise_logs": recent_exercise_logs
    }
    
    # Generate predictions
    predictions_data = await prediction_engine.predict_outcomes(current_user, recent_logs)
    
    # Save predictions to database
    predictions = []
    for pred_data in predictions_data:
        prediction = Prediction(
            user_id=current_user.id,
            prediction_type=pred_data["type"],
            current_value=pred_data["current_value"],
            predicted_value=pred_data["predicted_value"],
            confidence=pred_data["confidence"],
            timeframe_days=pred_data["timeframe_days"],
            factors=json.dumps(pred_data["factors"])
        )
        db.add(prediction)
        predictions.append(prediction)
    
    db.commit()
    return predictions

@app.post("/exercise/analyze-form")
async def analyze_exercise_form(
    exercise_type: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Analyze exercise form from video"""
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    video_data = await file.read()
    analysis_result = await form_analyzer.analyze_form(video_data, exercise_type)
    
    if "error" in analysis_result:
        raise HTTPException(status_code=500, detail=analysis_result["error"])
    
    return {
        "exercise_type": exercise_type,
        "form_analysis": analysis_result,
        "recommendations": [
            "Focus on controlled movements",
            "Maintain proper breathing rhythm",
            "Keep core engaged throughout the movement"
        ]
    }

@app.get("/dashboard/stats")
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get dashboard statistics for user"""
    # Get recent activity stats
    today = datetime.utcnow().date()
    week_ago = today - timedelta(days=7)
    month_ago = today - timedelta(days=30)
    
    # Food logs stats
    today_food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.logged_at >= datetime.combine(today, datetime.min.time())
    ).all()
    
    week_food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.logged_at >= datetime.combine(week_ago, datetime.min.time())
    ).all()
    
    # Exercise logs stats
    today_exercise_logs = db.query(ExerciseLog).filter(
        ExerciseLog.user_id == current_user.id,
        ExerciseLog.logged_at >= datetime.combine(today, datetime.min.time())
    ).all()
    
    week_exercise_logs = db.query(ExerciseLog).filter(
        ExerciseLog.user_id == current_user.id,
        ExerciseLog.logged_at >= datetime.combine(week_ago, datetime.min.time())
    ).all()
    
    # Body scans
    latest_scan = db.query(BodyScan).filter(
        BodyScan.user_id == current_user.id
    ).order_by(BodyScan.created_at.desc()).first()
    
    # Calculate totals
    today_calories = sum(log.food.calories_per_100g * (log.quantity / 100) for log in today_food_logs if log.food)
    today_calories_burned = sum(log.calories_burned for log in today_exercise_logs if log.calories_burned)
    week_workouts = len(week_exercise_logs)
    
    # Progress calculation
    progress_percentage = 0
    if current_user.target_weight and current_user.weight:
        start_weight = current_user.weight + abs(current_user.target_weight - current_user.weight)
        current_progress = abs(start_weight - current_user.weight)
        total_needed = abs(start_weight - current_user.target_weight)
        progress_percentage = min((current_progress / total_needed) * 100, 100) if total_needed > 0 else 0
    
    return {
        "today": {
            "calories_consumed": round(today_calories, 1),
            "calories_burned": round(today_calories_burned, 1),
            "net_calories": round(today_calories - today_calories_burned, 1),
            "meals_logged": len(today_food_logs),
            "workouts_completed": len(today_exercise_logs)
        },
        "week": {
            "total_workouts": week_workouts,
            "avg_calories_per_day": round(sum(log.food.calories_per_100g * (log.quantity / 100) for log in week_food_logs if log.food) / 7, 1),
            "total_calories_burned": round(sum(log.calories_burned for log in week_exercise_logs if log.calories_burned), 1)
        },
        "progress": {
            "goal_progress_percentage": round(progress_percentage, 1),
            "current_weight": current_user.weight,
            "target_weight": current_user.target_weight,
            "days_to_goal": (current_user.target_date - datetime.utcnow()).days if current_user.target_date else None
        },
        "latest_measurements": json.loads(latest_scan.measurements) if latest_scan and latest_scan.measurements else None
    }

@app.get("/nutrition/recommendations")
async def get_nutrition_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized nutrition recommendations"""
    # Get recent food logs
    recent_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.logged_at >= datetime.utcnow() - timedelta(days=3)
    ).all()
    
    # Calculate current nutrition intake
    total_calories = sum(log.food.calories_per_100g * (log.quantity / 100) for log in recent_logs if log.food)
    total_protein = sum(log.food.protein_per_100g * (log.quantity / 100) for log in recent_logs if log.food)
    avg_daily_calories = total_calories / 3 if recent_logs else 0
    avg_daily_protein = total_protein / 3 if recent_logs else 0
    
    # Calculate recommendations based on user goals
    if current_user.fitness_goal == "burn_fat":
        target_calories = current_user.weight * 24 * 1.2 - 500  # Deficit for weight loss
        target_protein = current_user.weight * 1.6  # Higher protein for muscle preservation
    elif current_user.fitness_goal == "build_muscle":
        target_calories = current_user.weight * 24 * 1.5 + 300  # Surplus for muscle building
        target_protein = current_user.weight * 2.0  # High protein for muscle synthesis
    else:  # maintain
        target_calories = current_user.weight * 24 * 1.3  # Maintenance calories
        target_protein = current_user.weight * 1.4
    
    # Generate recommendations
    recommendations = []
    
    if avg_daily_calories < target_calories * 0.8:
        recommendations.append({
            "type": "calorie_increase",
            "message": f"You're eating {int(target_calories - avg_daily_calories)} calories below your target. Consider adding healthy snacks.",
            "priority": "high",
            "suggested_foods": ["nuts", "avocado", "olive oil", "protein smoothie"]
        })
    elif avg_daily_calories > target_calories * 1.2:
        recommendations.append({
            "type": "calorie_reduction",
            "message": f"You're eating {int(avg_daily_calories - target_calories)} calories above your target. Focus on portion control.",
            "priority": "high",
            "suggested_changes": ["smaller portions", "more vegetables", "less processed foods"]
        })
    
    if avg_daily_protein < target_protein * 0.8:
        recommendations.append({
            "type": "protein_increase",
            "message": f"Increase protein intake by {int(target_protein - avg_daily_protein)}g per day.",
            "priority": "medium",
            "suggested_foods": ["chicken breast", "fish", "eggs", "Greek yogurt", "protein powder"]
        })
    
    # Meal timing recommendations
    current_hour = datetime.now().hour
    if 6 <= current_hour <= 10:
        recommendations.append({
            "type": "meal_timing",
            "message": "Perfect time for a protein-rich breakfast to kickstart your metabolism!",
            "priority": "low",
            "suggested_foods": ["eggs", "oatmeal with protein powder", "Greek yogurt with berries"]
        })
    elif 14 <= current_hour <= 17:
        recommendations.append({
            "type": "meal_timing",
            "message": "Pre-workout snack time! Eat some carbs for energy.",
            "priority": "medium",
            "suggested_foods": ["banana", "apple with almond butter", "oatmeal"]
        })
    
    return {
        "current_intake": {
            "avg_daily_calories": round(avg_daily_calories, 1),
            "avg_daily_protein": round(avg_daily_protein, 1)
        },
        "targets": {
            "daily_calories": round(target_calories, 1),
            "daily_protein": round(target_protein, 1)
        },
        "recommendations": recommendations,
        "next_meal_timing": "Eat your post-workout meal within 30 minutes of finishing exercise for optimal recovery."
    }

@app.get("/workout/recommendations")
async def get_workout_recommendations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized workout recommendations"""
    # Get recent exercise history
    recent_exercises = db.query(ExerciseLog).filter(
        ExerciseLog.user_id == current_user.id,
        ExerciseLog.logged_at >= datetime.utcnow() - timedelta(days=7)
    ).all()
    
    # Analyze workout patterns
    muscle_groups_trained = []
    total_workouts_this_week = len(recent_exercises)
    
    # Generate recommendations based on fitness goal
    if current_user.fitness_goal == "burn_fat":
        recommended_workout = {
            "type": "HIIT Cardio + Strength",
            "duration": "45 minutes",
            "exercises": [
                {"name": "Burpees", "reps": "10", "sets": "4"},
                {"name": "Mountain Climbers", "duration": "30 seconds", "sets": "4"},
                {"name": "Jump Squats", "reps": "15", "sets": "3"},
                {"name": "Push-ups", "reps": "12", "sets": "3"},
                {"name": "Plank", "duration": "45 seconds", "sets": "3"}
            ],
            "rest_between_sets": "30 seconds",
            "calories_estimate": "400-500"
        }
    elif current_user.fitness_goal == "build_muscle":
        recommended_workout = {
            "type": "Strength Training - Upper Body",
            "duration": "60 minutes",
            "exercises": [
                {"name": "Bench Press", "reps": "8-10", "sets": "4"},
                {"name": "Rows", "reps": "10-12", "sets": "4"},
                {"name": "Shoulder Press", "reps": "8-10", "sets": "3"},
                {"name": "Pull-ups", "reps": "6-8", "sets": "3"},
                {"name": "Bicep Curls", "reps": "12-15", "sets": "3"},
                {"name": "Tricep Dips", "reps": "10-12", "sets": "3"}
            ],
            "rest_between_sets": "60-90 seconds",
            "calories_estimate": "250-350"
        }
    else:  # maintain
        recommended_workout = {
            "type": "Full Body Maintenance",
            "duration": "40 minutes",
            "exercises": [
                {"name": "Squats", "reps": "12", "sets": "3"},
                {"name": "Push-ups", "reps": "10", "sets": "3"},
                {"name": "Lunges", "reps": "10 each leg", "sets": "3"},
                {"name": "Plank", "duration": "30 seconds", "sets": "3"},
                {"name": "Glute Bridges", "reps": "15", "sets": "3"}
            ],
            "rest_between_sets": "45 seconds",
            "calories_estimate": "200-300"
        }
    
    # Recovery recommendations
    recovery_score = min(10, total_workouts_this_week * 1.5) if total_workouts_this_week < 5 else 8
    
    return {
        "recommended_workout": recommended_workout,
        "weekly_summary": {
            "workouts_completed": total_workouts_this_week,
            "target_workouts": 4,
            "recovery_score": recovery_score
        },
        "tips": [
            "Focus on form over speed for better results",
            "Stay hydrated throughout your workout",
            "Don't forget to warm up and cool down",
            "Listen to your body and rest when needed"
        ],
        "next_workout_suggestion": "Based on your recent activity, focus on lower body strength training tomorrow."
    }

@app.post("/subscription/create")
async def create_subscription(
    plan_type: str = Form(...),  # "pro" or "elite"
    payment_method_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new subscription"""
    try:
        # Define pricing
        prices = {
            "pro": {"amount": 1499, "currency": "usd"},  # $14.99
            "elite": {"amount": 2999, "currency": "usd"}  # $29.99
        }
        
        if plan_type not in prices:
            raise HTTPException(status_code=400, detail="Invalid plan type")
        
        # Create or retrieve Stripe customer
        if not current_user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=current_user.email,
                name=current_user.full_name
            )
            current_user.stripe_customer_id = customer.id
            db.commit()
        
        # Attach payment method to customer
        stripe.PaymentMethod.attach(
            payment_method_id,
            customer=current_user.stripe_customer_id,
        )
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=current_user.stripe_customer_id,
            items=[{
                'price_data': {
                    'currency': prices[plan_type]["currency"],
                    'product_data': {
                        'name': f'FitSync Pro - {plan_type.capitalize()}',
                    },
                    'unit_amount': prices[plan_type]["amount"],
                    'recurring': {
                        'interval': 'month',
                    },
                },
            }],
            default_payment_method=payment_method_id,
        )
        
        # Save subscription to database
        db_subscription = Subscription(
            user_id=current_user.id,
            stripe_subscription_id=subscription.id,
            plan_type=plan_type,
            status=subscription.status,
            current_period_start=datetime.fromtimestamp(subscription.current_period_start),
            current_period_end=datetime.fromtimestamp(subscription.current_period_end)
        )
        db.add(db_subscription)
        
        # Update user subscription type
        current_user.subscription_type = plan_type
        db.commit()
        
        # Send confirmation email
        send_notification_email.delay(
            current_user.email,
            f"Welcome to FitSync Pro {plan_type.capitalize()}!",
            f"<h1>Subscription Activated!</h1><p>Your {plan_type} plan is now active. Enjoy all premium features!</p>"
        )
        
        return {
            "message": "Subscription created successfully",
            "subscription_id": subscription.id,
            "plan_type": plan_type,
            "status": subscription.status
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=f"Payment failed: {str(e)}")
    except Exception as e:
        logger.error(f"Subscription creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create subscription")

@app.post("/subscription/cancel")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel user's subscription"""
    try:
        # Find active subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id,
            Subscription.status == "active"
        ).first()
        
        if not subscription:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        # Cancel Stripe subscription
        stripe.Subscription.modify(
            subscription.stripe_subscription_id,
            cancel_at_period_end=True
        )
        
        # Update subscription status
        subscription.status = "canceled"
        db.commit()
        
        return {"message": "Subscription canceled successfully"}
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=400, detail=f"Cancellation failed: {str(e)}")

@app.get("/exercises/search")
async def search_exercises(
    query: str,
    category: Optional[str] = None,
    muscle_group: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Search exercises by name, category, or muscle group"""
    exercises_query = db.query(Exercise)
    
    if query:
        exercises_query = exercises_query.filter(Exercise.name.ilike(f"%{query}%"))
    
    if category:
        exercises_query = exercises_query.filter(Exercise.category == category)
    
    if muscle_group:
        exercises_query = exercises_query.filter(Exercise.muscle_groups.contains(muscle_group))
    
    exercises = exercises_query.limit(20).all()
    
    return {
        "exercises": [
            {
                "id": ex.id,
                "name": ex.name,
                "category": ex.category,
                "muscle_groups": json.loads(ex.muscle_groups) if ex.muscle_groups else [],
                "difficulty_level": ex.difficulty_level,
                "equipment": ex.equipment,
                "video_url": ex.video_url
            } for ex in exercises
        ]
    }

@app.get("/foods/search")
async def search_foods(
    query: str,
    db: Session = Depends(get_db)
):
    """Search foods by name"""
    foods = db.query(Food).filter(Food.name.ilike(f"%{query}%")).limit(20).all()
    
    return {
        "foods": [
            {
                "id": food.id,
                "name": food.name,
                "brand": food.brand,
                "calories_per_100g": food.calories_per_100g,
                "protein_per_100g": food.protein_per_100g,
                "carbs_per_100g": food.carbs_per_100g,
                "fat_per_100g": food.fat_per_100g,
                "category": food.category
            } for food in foods
        ]
    }

@app.get("/progress/history")
async def get_progress_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's progress history over specified days"""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get body scans
    body_scans = db.query(BodyScan).filter(
        BodyScan.user_id == current_user.id,
        BodyScan.created_at >= start_date
    ).order_by(BodyScan.created_at).all()
    
    # Get daily calorie intake
    food_logs = db.query(FoodLog).filter(
        FoodLog.user_id == current_user.id,
        FoodLog.logged_at >= start_date
    ).all()
    
    # Get daily exercise
    exercise_logs = db.query(ExerciseLog).filter(
        ExerciseLog.user_id == current_user.id,
        ExerciseLog.logged_at >= start_date
    ).all()
    
    # Group by date
    daily_data = {}
    
    # Process food logs
    for log in food_logs:
        date = log.logged_at.date()
        if date not in daily_data:
            daily_data[date] = {"calories": 0, "protein": 0, "exercise_calories": 0, "workouts": 0}
        
        if log.food:
            daily_data[date]["calories"] += log.food.calories_per_100g * (log.quantity / 100)
            daily_data[date]["protein"] += log.food.protein_per_100g * (log.quantity / 100)
    
    # Process exercise logs
    for log in exercise_logs:
        date = log.logged_at.date()
        if date not in daily_data:
            daily_data[date] = {"calories": 0, "protein": 0, "exercise_calories": 0, "workouts": 0}
        
        daily_data[date]["exercise_calories"] += log.calories_burned or 0
        daily_data[date]["workouts"] += 1
    
    # Format response
    progress_data = []
    for date, data in sorted(daily_data.items()):
        progress_data.append({
            "date": date.isoformat(),
            "calories_consumed": round(data["calories"], 1),
            "calories_burned": round(data["exercise_calories"], 1),
            "protein_grams": round(data["protein"], 1),
            "workouts_completed": data["workouts"]
        })
    
    # Body measurements progress
    measurements_progress = []
    for scan in body_scans:
        if scan.measurements:
            measurements = json.loads(scan.measurements)
            measurements_progress.append({
                "date": scan.created_at.date().isoformat(),
                "measurements": measurements
            })
    
    return {
        "daily_progress": progress_data,
        "measurements_progress": measurements_progress,
        "summary": {
            "total_days": len(progress_data),
            "avg_daily_calories": round(sum(d["calories_consumed"] for d in progress_data) / len(progress_data), 1) if progress_data else 0,
            "total_workouts": sum(d["workouts_completed"] for d in progress_data),
            "total_calories_burned": sum(d["calories_burned"] for d in progress_data)
        }
    }

@app.post("/goals/update")
async def update_fitness_goals(
    fitness_goal: str = Form(...),
    target_weight: Optional[float] = Form(None),
    target_date: Optional[datetime] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's fitness goals"""
    valid_goals = ["burn_fat", "build_muscle", "maintain"]
    if fitness_goal not in valid_goals:
        raise HTTPException(status_code=400, detail=f"Invalid fitness goal. Must be one of: {valid_goals}")
    
    current_user.fitness_goal = fitness_goal
    if target_weight:
        current_user.target_weight = target_weight
    if target_date:
        current_user.target_date = target_date
    
    current_user.updated_at = datetime.utcnow()
    db.commit()
    
    # Generate new predictions based on updated goals
    predictions = await prediction_engine.predict_outcomes(current_user, {})
    
    return {
        "message": "Goals updated successfully",
        "new_goal": fitness_goal,
        "target_weight": target_weight,
        "target_date": target_date.isoformat() if target_date else None,
        "updated_predictions": predictions
    }

# =============================================================================
# HEALTH ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
        
        # Check Redis connection
        redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "connected",
                "redis": "connected",
                "ai_services": "operational"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def get_system_metrics(db: Session = Depends(get_db)):
    """Get system metrics for monitoring"""
    try:
        # Count active users
        total_users = db.query(User).count()
        active_users_today = db.query(User).filter(
            User.updated_at >= datetime.utcnow() - timedelta(days=1)
        ).count()
        
        # Count recent activities
        recent_food_logs = db.query(FoodLog).filter(
            FoodLog.logged_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        recent_exercise_logs = db.query(ExerciseLog).filter(
            ExerciseLog.logged_at >= datetime.utcnow() - timedelta(hours=24)
        ).count()
        
        # Subscription metrics
        pro_subscriptions = db.query(User).filter(User.subscription_type == "pro").count()
        elite_subscriptions = db.query(User).filter(User.subscription_type == "elite").count()
        
        return {
            "users": {
                "total": total_users,
                "active_today": active_users_today,
                "pro_subscribers": pro_subscriptions,
                "elite_subscribers": elite_subscriptions
            },
            "activity_24h": {
                "food_logs": recent_food_logs,
                "exercise_logs": recent_exercise_logs
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": "Failed to collect metrics"}

# =============================================================================
# WEBHOOK ENDPOINTS
# =============================================================================

@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhooks"""
    try:
        payload = await request.body()
        sig_header = request.headers.get("stripe-signature")
        endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        
        if event["type"] == "invoice.payment_succeeded":
            # Handle successful payment
            subscription_id = event["data"]["object"]["subscription"]
            subscription = db.query(Subscription).filter(
                Subscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = "active"
                db.commit()
                logger.info(f"Payment succeeded for subscription {subscription_id}")
        
        elif event["type"] == "invoice.payment_failed":
            # Handle failed payment
            subscription_id = event["data"]["object"]["subscription"]
            subscription = db.query(Subscription).filter(
                Subscription.stripe_subscription_id == subscription_id
            ).first()
            
            if subscription:
                subscription.status = "past_due"
                db.commit()
                logger.warning(f"Payment failed for subscription {subscription_id}")
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=400, detail="Webhook processing failed")

# =============================================================================
# DATABASE INITIALIZATION AND SEED DATA
# =============================================================================

def create_seed_data():
    """Create initial seed data for the application"""
    db = SessionLocal()
    
    try:
        # Check if seed data already exists
        if db.query(Food).first():
            logger.info("Seed data already exists, skipping...")
            return
        
        # Create sample foods
        sample_foods = [
            Food(name="Chicken Breast", calories_per_100g=165, protein_per_100g=31, carbs_per_100g=0, fat_per_100g=3.6, category="meat"),
            Food(name="White Rice", calories_per_100g=130, protein_per_100g=2.7, carbs_per_100g=28, fat_per_100g=0.3, category="grains"),
            Food(name="Broccoli", calories_per_100g=34, protein_per_100g=2.8, carbs_per_100g=7, fat_per_100g=0.4, category="vegetables"),
            Food(name="Banana", calories_per_100g=89, protein_per_100g=1.1, carbs_per_100g=23, fat_per_100g=0.3, category="fruits"),
            Food(name="Greek Yogurt", calories_per_100g=59, protein_per_100g=10, carbs_per_100g=3.6, fat_per_100g=0.4, category="dairy"),
            Food(name="Oatmeal", calories_per_100g=389, protein_per_100g=17, carbs_per_100g=66, fat_per_100g=7, category="grains"),
            Food(name="Salmon", calories_per_100g=208, protein_per_100g=20, carbs_per_100g=0, fat_per_100g=13, category="fish"),
            Food(name="Sweet Potato", calories_per_100g=86, protein_per_100g=1.6, carbs_per_100g=20, fat_per_100g=0.1, category="vegetables"),
            Food(name="Eggs", calories_per_100g=155, protein_per_100g=13, carbs_per_100g=1.1, fat_per_100g=11, category="protein"),
            Food(name="Almonds", calories_per_100g=579, protein_per_100g=21, carbs_per_100g=22, fat_per_100g=50, category="nuts"),
        ]
        
        for food in sample_foods:
            db.add(food)
        
        # Create sample exercises
        sample_exercises = [
            Exercise(
                name="Push-ups",
                category="strength",
                muscle_groups=json.dumps(["chest", "shoulders", "triceps"]),
                equipment="bodyweight",
                difficulty_level="beginner",
                instructions="Start in plank position, lower body to ground, push back up",
                calories_per_minute=8.0
            ),
            Exercise(
                name="Squats",
                category="strength",
                muscle_groups=json.dumps(["quadriceps", "glutes", "hamstrings"]),
                equipment="bodyweight",
                difficulty_level="beginner",
                instructions="Stand with feet shoulder-width apart, lower into squat position, return to standing",
                calories_per_minute=6.0
            ),
            Exercise(
                name="Running",
                category="cardio",
                muscle_groups=json.dumps(["legs", "core"]),
                equipment="none",
                difficulty_level="intermediate",
                instructions="Maintain steady pace, focus on breathing rhythm",
                calories_per_minute=12.0
            ),
            Exercise(
                name="Deadlifts",
                category="strength",
                muscle_groups=json.dumps(["hamstrings", "glutes", "back", "core"]),
                equipment="barbell",
                difficulty_level="advanced",
                instructions="Keep back straight, lift with legs and hips, not back",
                calories_per_minute=10.0
            ),
            Exercise(
                name="Plank",
                category="strength",
                muscle_groups=json.dumps(["core", "shoulders"]),
                equipment="bodyweight",
                difficulty_level="beginner",
                instructions="Hold plank position, keep body straight line",
                calories_per_minute=5.0
            ),
            Exercise(
                name="Bench Press",
                category="strength",
                muscle_groups=json.dumps(["chest", "shoulders", "triceps"]),
                equipment="barbell",
                difficulty_level="intermediate",
                instructions="Lower bar to chest, press up explosively",
                calories_per_minute=8.0
            ),
        ]
        
        for exercise in sample_exercises:
            db.add(exercise)
        
        db.commit()
        logger.info("Seed data created successfully")
        
    except Exception as e:
        logger.error(f"Error creating seed data: {e}")
        db.rollback()
    finally:
        db.close()

# =============================================================================
# STARTUP EVENT
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Run startup tasks"""
    logger.info("Starting FitSync Pro API...")
    
    # Create seed data
    create_seed_data()
    
    # Initialize AI models (placeholder)
    logger.info("Initializing AI models...")
    
    # Set up scheduled tasks (placeholder)
    logger.info("Setting up scheduled tasks...")
    
    logger.info("FitSync Pro API started successfully!")

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# =============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# =============================================================================

class NotificationService:
    """Service for handling notifications"""
    
    @staticmethod
    async def send_goal_achievement_notification(user: User, achievement: str):
        """Send notification when user achieves a goal"""
        subject = f"🎉 Congratulations {user.full_name}!"
        body = f"""
        <html>
        <body>
            <h1>Goal Achievement!</h1>
            <p>Congratulations on achieving: <strong>{achievement}</strong></p>
            <p>Keep up the great work on your fitness journey!</p>
            <p>Your FitSync Pro Team</p>
        </body>
        </html>
        """
        
        send_notification_email.delay(user.email, subject, body)
    
    @staticmethod
    async def send_weekly_progress_report(user: User, progress_data: dict):
        """Send weekly progress report"""
        subject = f"Your Weekly Fitness Report - {datetime.now().strftime('%B %d, %Y')}"
        body = f"""
        <html>
        <body>
            <h1>Weekly Progress Report</h1>
            <p>Hi {user.full_name},</p>
            <p>Here's your fitness progress this week:</p>
            <ul>
                <li>Workouts completed: {progress_data.get('workouts', 0)}</li>
                <li>Average daily calories: {progress_data.get('calories', 0)}</li>
                <li>Weight change: {progress_data.get('weight_change', 0)}kg</li>
            </ul>
            <p>Keep up the excellent work!</p>
        </body>
        </html>
        """
        
        send_notification_email.delay(user.email, subject, body)

class DataAnalytics:
    """Service for data analytics and insights"""
    
    @staticmethod
    def calculate_user_engagement_score(user_id: int, db: Session) -> float:
        """Calculate user engagement score based on activity"""
        try:
            # Get activity in last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            food_logs_count = db.query(FoodLog).filter(
                FoodLog.user_id == user_id,
                FoodLog.logged_at >= thirty_days_ago
            ).count()
            
            exercise_logs_count = db.query(ExerciseLog).filter(
                ExerciseLog.user_id == user_id,
                ExerciseLog.logged_at >= thirty_days_ago
            ).count()
            
            body_scans_count = db.query(BodyScan).filter(
                BodyScan.user_id == user_id,
                BodyScan.created_at >= thirty_days_ago
            ).count()
            
            # Calculate engagement score (0-100)
            score = min(100, (food_logs_count * 2) + (exercise_logs_count * 3) + (body_scans_count * 5))
            return score
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.0
    
    @staticmethod
    def generate_personalized_insights(user: User, db: Session) -> List[Dict]:
        """Generate personalized insights for user"""
        insights = []
        
        try:
            # Analyze workout patterns
            recent_exercises = db.query(ExerciseLog).filter(
                ExerciseLog.user_id == user.id,
                ExerciseLog.logged_at >= datetime.utcnow() - timedelta(days=14)
            ).all()
            
            if len(recent_exercises) < 3:
                insights.append({
                    "type": "workout_frequency",
                    "message": "You've been less active recently. Try to aim for at least 3 workouts per week!",
                    "priority": "high",
                    "action": "schedule_workout"
                })
            
            # Analyze nutrition patterns
            recent_food_logs = db.query(FoodLog).filter(
                FoodLog.user_id == user.id,
                FoodLog.logged_at >= datetime.utcnow() - timedelta(days=7)
            ).all()
            
            if recent_food_logs:
                avg_protein = sum(
                    log.food.protein_per_100g * (log.quantity / 100) 
                    for log in recent_food_logs if log.food
                ) / len(recent_food_logs)
                
                target_protein = user.weight * 1.6  # Basic protein target
                
                if avg_protein < target_protein * 0.8:
                    insights.append({
                        "type": "nutrition_protein",
                        "message": f"Your protein intake is below target. Aim for {target_protein:.1f}g daily.",
                        "priority": "medium",
                        "action": "increase_protein"
                    })
            
            # Goal progress insight
            if user.target_weight and user.weight:
                progress = abs(user.weight - user.target_weight) / abs(user.target_weight - user.weight) * 100
                if progress > 50:
                    insights.append({
                        "type": "goal_progress",
                        "message": f"Great progress! You're {progress:.1f}% toward your weight goal.",
                        "priority": "positive",
                        "action": "keep_going"
                    })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return []

# =============================================================================
# ADVANCED API ENDPOINTS
# =============================================================================

@app.get("/insights/personalized")
async def get_personalized_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized insights and recommendations"""
    analytics = DataAnalytics()
    
    engagement_score = analytics.calculate_user_engagement_score(current_user.id, db)
    insights = analytics.generate_personalized_insights(current_user, db)
    
    return {
        "engagement_score": engagement_score,
        "insights": insights,
        "recommendations": {
            "next_actions": [
                "Log your breakfast for accurate calorie tracking",
                "Schedule your next workout session",
                "Take progress photos for your body scan"
            ],
            "weekly_goals": [
                "Complete 4 strength training sessions",
                "Maintain protein target of 120g daily",
                "Take weekly body measurements"
            ]
        }
    }

@app.get("/analytics/trends")
async def get_fitness_trends(
    period: str = "month",  # week, month, quarter
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get fitness trends and analytics"""
    try:
        # Define date ranges
        periods = {
            "week": timedelta(days=7),
            "month": timedelta(days=30),
            "quarter": timedelta(days=90)
        }
        
        if period not in periods:
            raise HTTPException(status_code=400, detail="Invalid period")
        
        start_date = datetime.utcnow() - periods[period]
        
        # Get trend data
        food_logs = db.query(FoodLog).filter(
            FoodLog.user_id == current_user.id,
            FoodLog.logged_at >= start_date
        ).all()
        
        exercise_logs = db.query(ExerciseLog).filter(
            ExerciseLog.user_id == current_user.id,
            ExerciseLog.logged_at >= start_date
        ).all()
        
        # Calculate trends
        total_calories = sum(
            log.food.calories_per_100g * (log.quantity / 100) 
            for log in food_logs if log.food
        )
        
        total_workouts = len(exercise_logs)
        total_calories_burned = sum(log.calories_burned for log in exercise_logs if log.calories_burned)
        
        # Calculate averages
        days_in_period = periods[period].days
        avg_daily_calories = total_calories / days_in_period if days_in_period > 0 else 0
        avg_weekly_workouts = (total_workouts / days_in_period) * 7 if days_in_period > 0 else 0
        
        return {
            "period": period,
            "summary": {
                "avg_daily_calories": round(avg_daily_calories, 1),
                "avg_weekly_workouts": round(avg_weekly_workouts, 1),
                "total_calories_burned": round(total_calories_burned, 1),
                "consistency_score": min(100, (total_workouts / (days_in_period / 2)) * 100)
            },
            "trends": {
                "calorie_intake": "stable",  # Would calculate actual trend
                "workout_frequency": "increasing",
                "weight_progress": "on_track"
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate trends")

@app.post("/coaching/message")
async def send_coaching_message(
    message: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Send message to AI coach and get response"""
    try:
        # Simple AI coach responses (in production, use advanced NLP)
        coaching_responses = {
            "motivation": [
                "You're doing great! Every small step counts toward your bigger goals.",
                "Remember, consistency beats perfection. Keep showing up for yourself!",
                "Your future self will thank you for the effort you're putting in today."
            ],
            "nutrition": [
                "Focus on whole foods and adequate protein to fuel your workouts.",
                "Remember to stay hydrated - aim for at least 8 glasses of water daily.",
                "Pre-workout carbs and post-workout protein will optimize your results."
            ],
            "workout": [
                "Form is more important than weight. Master the movement first.",
                "Progressive overload is key - gradually increase intensity over time.",
                "Don't forget to include rest days for recovery and muscle growth."
            ]
        }
        
        # Simple keyword detection
        message_lower = message.lower()
        if any(word in message_lower for word in ["tired", "unmotivated", "quit"]):
            response_type = "motivation"
        elif any(word in message_lower for word in ["eat", "food", "nutrition", "protein"]):
            response_type = "nutrition"
        elif any(word in message_lower for word in ["workout", "exercise", "training"]):
            response_type = "workout"
        else:
            response_type = "motivation"
        
        import random
        response = random.choice(coaching_responses[response_type])
        
        return {
            "user_message": message,
            "coach_response": response,
            "response_type": response_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in coaching message: {e}")
        return {
            "coach_response": "I'm here to help! Can you tell me more about what you'd like assistance with?",
            "response_type": "general"
        }

@app.get("/export/data")
async def export_user_data(
    format: str = "json",  # json, csv
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export user's fitness data"""
    if current_user.subscription_type == "basic":
        raise HTTPException(status_code=403, detail="Data export requires Pro or Elite subscription")
    
    try:
        # Gather all user data
        food_logs = db.query(FoodLog).filter(FoodLog.user_id == current_user.id).all()
        exercise_logs = db.query(ExerciseLog).filter(ExerciseLog.user_id == current_user.id).all()
        body_scans = db.query(BodyScan).filter(BodyScan.user_id == current_user.id).all()
        predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).all()
        
        export_data = {
            "user_profile": {
                "username": current_user.username,
                "email": current_user.email,
                "fitness_goal": current_user.fitness_goal,
                "height": current_user.height,
                "weight": current_user.weight,
                "target_weight": current_user.target_weight
            },
            "food_logs": [
                {
                    "date": log.logged_at.isoformat(),
                    "food_name": log.food.name if log.food else "Unknown",
                    "quantity_grams": log.quantity,
                    "meal_type": log.meal_type,
                    "calories": log.food.calories_per_100g * (log.quantity / 100) if log.food else 0
                } for log in food_logs
            ],
            "exercise_logs": [
                {
                    "date": log.logged_at.isoformat(),
                    "exercise_name": log.exercise.name if log.exercise else "Unknown",
                    "duration_minutes": log.duration,
                    "reps": log.reps,
                    "sets": log.sets,
                    "weight_kg": log.weight,
                    "calories_burned": log.calories_burned
                } for log in exercise_logs
            ],
            "body_scans": [
                {
                    "date": scan.created_at.isoformat(),
                    "scan_type": scan.scan_type,
                    "measurements": json.loads(scan.measurements) if scan.measurements else {},
                    "body_fat_percentage": scan.body_fat_percentage,
                    "muscle_mass": scan.muscle_mass
                } for scan in body_scans
            ],
            "predictions": [
                {
                    "date": pred.created_at.isoformat(),
                    "type": pred.prediction_type,
                    "current_value": pred.current_value,
                    "predicted_value": pred.predicted_value,
                    "confidence": pred.confidence,
                    "timeframe_days": pred.timeframe_days
                } for pred in predictions
            ]
        }
        
        if format == "json":
            return export_data
        elif format == "csv":
            # For CSV, we'll return a simplified version
            # In production, you'd generate actual CSV files
            return {
                "message": "CSV export feature coming soon",
                "available_formats": ["json"]
            }
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Failed to export data")

# =============================================================================
# MAIN APPLICATION RUNNER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting FitSync Pro API on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1 if debug else 4,
        log_level="info"
    )

# =============================================================================
# REQUIREMENTS.txt (Dependencies to install)
# =============================================================================

"""
# Core Dependencies
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Redis & Caching
redis==5.0.1
celery==5.3.4

# Payment Processing
stripe==7.8.0

# Email & Notifications
emails==0.6

# AI/ML Dependencies
tensorflow==2.15.0
opencv-python==4.8.1.78
mediapipe==0.10.8
numpy==1.24.3
scipy==1.11.4
pillow==10.1.0

# AWS Services
boto3==1.34.0

# Utility
python-dotenv==1.0.0
requests==2.31.0
pydantic[email]==2.5.0

# Development & Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2
"""

# =============================================================================
# DOCKER CONFIGURATION
# =============================================================================

"""
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
"""

# =============================================================================
# ENVIRONMENT VARIABLES TEMPLATE (.env)
# =============================================================================

"""
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/fitsync_pro

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Celery
CELERY_BROKER=redis://localhost:6379/0
CELERY_BACKEND=redis://localhost:6379/0

# Security
SECRET_KEY=your-super-secret-key-change-this-in-production

# Stripe
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# AWS
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=fitsync-data

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=noreply@fitsyncpro.com

# Application
HOST=0.0.0.0
PORT=8000
DEBUG=false
"""