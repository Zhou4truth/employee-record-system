from pydantic import BaseModel
from typing import List, Optional, Union, Any
from datetime import datetime

# Admin schemas
class AdminBase(BaseModel):
    username: str

class AdminCreate(AdminBase):
    password: str

class Admin(AdminBase):
    id: int
    is_active: bool

    class Config:
        orm_mode = True

# Employee schemas
class EmployeeBase(BaseModel):
    name: str
    gender: str
    department: str
    position: str
    employee_number: str

class EmployeeCreate(EmployeeBase):
    pass

class Employee(EmployeeBase):
    id: int
    registered_at: datetime
    image_path: Optional[str] = None

    class Config:
        orm_mode = True

# Access record schemas
class AccessRecordBase(BaseModel):
    employee_id: int
    status: str

class AccessRecord(AccessRecordBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True

# Authentication schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Recognition result schema
class RecognitionResult(BaseModel):
    recognized: bool
    employee: Optional[Employee] = None
    message: str

# System status schema
class SystemStatus(BaseModel):
    camera_connected: bool
    door_status: str
    recognition_model: str
    accuracy: float
    last_maintenance: datetime
