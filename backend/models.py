from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from database import Base

class Admin(Base):
    __tablename__ = "admins"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    gender = Column(String)
    department = Column(String, index=True)
    position = Column(String)
    employee_number = Column(String, unique=True, index=True)
    face_encoding = Column(JSON)  # Store face encoding as JSON
    image_path = Column(String)
    registered_at = Column(DateTime, default=datetime.now)
    
    # Relationship with access records
    access_records = relationship("AccessRecord", back_populates="employee")

class AccessRecord(Base):
    __tablename__ = "access_records"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"))
    timestamp = Column(DateTime, default=datetime.now)
    status = Column(String)  # "granted" or "denied"
    
    # Relationship with employee
    employee = relationship("Employee", back_populates="access_records")

class SystemSettings(Base):
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String)
    door_id = Column(String)
    recognition_threshold = Column(Float, default=0.6)
    last_maintenance = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)
