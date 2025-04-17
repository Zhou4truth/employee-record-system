from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
import cv2
import numpy as np
import face_recognition
import base64
import os
import shutil
from datetime import datetime, timedelta

from database import SessionLocal, engine
import models
import schemas
from auth import create_access_token, get_current_user, get_password_hash, verify_password

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Employee Recognition System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify the exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication endpoints
@app.post("/login", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.Admin).filter(models.Admin.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Employee management endpoints
@app.post("/employees/", response_model=schemas.Employee)
async def create_employee(
    name: str = Form(...),
    gender: str = Form(...),
    department: str = Form(...),
    position: str = Form(...),
    employee_number: str = Form(...),
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: schemas.AdminBase = Depends(get_current_user)
):
    # Check if employee number already exists
    db_employee = db.query(models.Employee).filter(models.Employee.employee_number == employee_number).first()
    if db_employee:
        raise HTTPException(status_code=400, detail="Employee number already registered")
    
    # Process and save the face image
    contents = await face_image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces in the image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if not face_encodings:
        raise HTTPException(status_code=400, detail="Could not encode the face")
    
    face_encoding = face_encodings[0]
    
    # Save the image to a file
    os.makedirs("employee_images", exist_ok=True)
    image_path = f"employee_images/{employee_number}.jpg"
    cv2.imwrite(image_path, img)
    
    # Create new employee
    db_employee = models.Employee(
        name=name,
        gender=gender,
        department=department,
        position=position,
        employee_number=employee_number,
        face_encoding=face_encoding.tolist(),
        image_path=image_path,
        registered_at=datetime.now()
    )
    
    db.add(db_employee)
    db.commit()
    db.refresh(db_employee)
    
    return db_employee

@app.get("/employees/", response_model=List[schemas.Employee])
async def read_employees(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: schemas.AdminBase = Depends(get_current_user)
):
    employees = db.query(models.Employee).offset(skip).limit(limit).all()
    return employees

@app.get("/employees/{employee_id}", response_model=schemas.Employee)
async def read_employee(
    employee_id: int, 
    db: Session = Depends(get_db),
    current_user: schemas.AdminBase = Depends(get_current_user)
):
    db_employee = db.query(models.Employee).filter(models.Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    return db_employee

@app.put("/employees/{employee_id}", response_model=schemas.Employee)
async def update_employee(
    employee_id: int,
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    employee_number: Optional[str] = Form(None),
    face_image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: schemas.AdminBase = Depends(get_current_user)
):
    db_employee = db.query(models.Employee).filter(models.Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Update fields if provided
    if name:
        db_employee.name = name
    if gender:
        db_employee.gender = gender
    if department:
        db_employee.department = department
    if position:
        db_employee.position = position
    if employee_number and employee_number != db_employee.employee_number:
        # Check if new employee number already exists
        existing = db.query(models.Employee).filter(models.Employee.employee_number == employee_number).first()
        if existing and existing.id != employee_id:
            raise HTTPException(status_code=400, detail="Employee number already in use")
        db_employee.employee_number = employee_number
    
    # Process new face image if provided
    if face_image:
        contents = await face_image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces in the image
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="Could not encode the face")
        
        face_encoding = face_encodings[0]
        
        # Save the image to a file
        image_path = f"employee_images/{db_employee.employee_number}.jpg"
        cv2.imwrite(image_path, img)
        
        db_employee.face_encoding = face_encoding.tolist()
        db_employee.image_path = image_path
    
    db.commit()
    db.refresh(db_employee)
    
    return db_employee

@app.delete("/employees/{employee_id}")
async def delete_employee(
    employee_id: int, 
    db: Session = Depends(get_db),
    current_user: schemas.AdminBase = Depends(get_current_user)
):
    db_employee = db.query(models.Employee).filter(models.Employee.id == employee_id).first()
    if db_employee is None:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    # Delete the employee image if it exists
    if db_employee.image_path and os.path.exists(db_employee.image_path):
        os.remove(db_employee.image_path)
    
    db.delete(db_employee)
    db.commit()
    
    return {"message": "Employee deleted successfully"}

# Face recognition endpoint
@app.post("/recognize", response_model=schemas.RecognitionResult)
async def recognize_face(
    face_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    contents = await face_image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces in the image
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    
    if not face_locations:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    
    # Get face encodings
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    if not face_encodings:
        raise HTTPException(status_code=400, detail="Could not encode the face")
    
    unknown_encoding = face_encodings[0]
    
    # Get all employees from database
    employees = db.query(models.Employee).all()
    
    # Compare with known faces
    for employee in employees:
        known_encoding = np.array(employee.face_encoding)
        # Compare faces with a tolerance (lower is more strict)
        match = face_recognition.compare_faces([known_encoding], unknown_encoding, tolerance=0.6)
        
        if match[0]:
            # Record access
            access_record = models.AccessRecord(
                employee_id=employee.id,
                timestamp=datetime.now(),
                status="granted"
            )
            db.add(access_record)
            db.commit()
            
            return {
                "recognized": True,
                "employee": employee,
                "message": f"Access granted for {employee.name}"
            }
    
    # No match found
    return {
        "recognized": False,
        "employee": None,
        "message": "Access denied: Face not recognized"
    }

# System management endpoints
@app.get("/system/status", response_model=schemas.SystemStatus)
async def get_system_status(current_user: schemas.AdminBase = Depends(get_current_user)):
    # In a real system, you would check camera connections, door status, etc.
    return {
        "camera_connected": True,
        "door_status": "closed",
        "recognition_model": "face_recognition 1.3.0",
        "accuracy": 0.95,
        "last_maintenance": datetime.now() - timedelta(days=7)
    }

# Initialize admin user if none exists
@app.on_event("startup")
async def startup_event():
    db = SessionLocal()
    try:
        # Check if admin exists
        admin = db.query(models.Admin).first()
        if not admin:
            # Create default admin
            hashed_password = get_password_hash("admin123")
            default_admin = models.Admin(
                username="admin",
                hashed_password=hashed_password,
                is_active=True
            )
            db.add(default_admin)
            db.commit()
            print("Default admin user created")
    finally:
        db.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
