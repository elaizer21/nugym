from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from passlib.context import CryptContext
from datetime import datetime

DATABASE_URL = "sqlite:///./fitsync_pro.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    height = Column(Float)
    weight = Column(Float)
    activity_level = Column(String)
    fitness_goal = Column(String)
    target_weight = Column(Float)
    target_date = Column(DateTime)
    subscription_type = Column(String, default="basic")
    stripe_customer_id = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    body_scans = relationship("BodyScan", back_populates="user")
    food_logs = relationship("FoodLog", back_populates="user")
    exercise_logs = relationship("ExerciseLog", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")

class BodyScan(Base):
    __tablename__ = "body_scans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scan_type = Column(String)
    measurements = Column(Text)
    scan_data_url = Column(String)
    photos_urls = Column(Text)
    body_fat_percentage = Column(Float)
    muscle_mass = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="body_scans")

class FoodLog(Base):
    __tablename__ = "food_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    food_name = Column(String)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fats = Column(Float)
    logged_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="food_logs")

class ExerciseLog(Base):
    __tablename__ = "exercise_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_name = Column(String)
    duration_minutes = Column(Float)
    calories_burned = Column(Float)
    logged_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="exercise_logs")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    target_weight = Column(Float)
    target_date = Column(DateTime)
    predicted_success = Column(Boolean, default=None)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")