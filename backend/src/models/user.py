from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class User(BaseModel):
    """
    Represents a user of the system
    """
    id: Optional[str] = None
    email: EmailStr = Field(..., description="User's email address")
    name: Optional[str] = Field(None, description="User's display name")
    created_at: Optional[datetime] = None
    last_login_at: Optional[datetime] = None
    preferences: Optional[Dict[str, Any]] = {}


class UserCreate(BaseModel):
    """
    Model for creating a new user
    """
    email: EmailStr
    name: Optional[str] = None
    password: str  # In a real implementation, this would be hashed


class UserUpdate(BaseModel):
    """
    Model for updating user information
    """
    name: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """
    Response model for user information
    """
    id: str
    email: EmailStr
    name: Optional[str]
    created_at: Optional[datetime]
    last_login_at: Optional[datetime]
    preferences: Optional[Dict[str, Any]]