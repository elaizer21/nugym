from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    activity_level: Optional[str] = None
    fitness_goal: Optional[str] = None
    target_weight: Optional[float] = None
    target_date: Optional[datetime] = None

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: EmailStr
    username: str
    full_name: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    height: Optional[float]
    weight: Optional[float]
    activity_level: Optional[str]
    fitness_goal: Optional[str]
    target_weight: Optional[float]
    target_date: Optional[datetime]
    subscription_type: Optional[str]
    is_active: bool

    class Config:
        orm_mode = True

class BodyScanCreate(BaseModel):
    scan_type: str
    measurements: Optional[str] = None
    scan_data_url: Optional[str] = None
    photos_urls: Optional[str] = None
    body_fat_percentage: Optional[float] = None
    muscle_mass: Optional[float] = None

class BodyScanResponse(BodyScanCreate):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class FoodLogCreate(BaseModel):
    food_name: str
    calories: float
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fats: Optional[float] = None
    logged_at: Optional[datetime] = None

class FoodLogResponse(FoodLogCreate):
    id: int
    user_id: int

    class Config:
        orm_mode = True

class ExerciseLogCreate(BaseModel):
    exercise_name: str
    duration_minutes: float
    calories_burned: Optional[float] = None
    logged_at: Optional[datetime] = None

class ExerciseLogResponse(ExerciseLogCreate):
    id: int
    user_id: int

    class Config:
        orm_mode = True

class PredictionCreate(BaseModel):
    target_weight: float
    target_date: datetime

class PredictionResponse(PredictionCreate):
    id: int
    user_id: int
    predicted_success: Optional[bool] = None

    class Config:
        orm_mode = True
