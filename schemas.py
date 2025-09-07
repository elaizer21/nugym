from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

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
        orm_mode = True

# More schemas (FoodLog, ExerciseLog, Prediction, etc.) can be added.