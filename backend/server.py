from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
from passlib.hash import bcrypt
import gspread
from google.oauth2.service_account import Credentials
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import requests

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT Configuration
JWT_SECRET = os.environ.get('JWT_SECRET', 'pena-digital-store-secret-key-2025')
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Security
security = HTTPBearer()

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Categories
CATEGORIES = {
    "1": "Perangkat Ajar Guru",
    "2": "Kunci Jawaban",
    "3": "Produk Lainnya"
}

# ==================== Models ====================

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    token: str
    username: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_name: str = "Pena Digital Store"
    logo_url: str = ""
    sheet_url_1: str = "https://docs.google.com/spreadsheets/d/1X8lIZGvzUW_mt_6qDayplpPGHD1nsK1xsaBNQy2ozc8/edit?usp=sharing"
    sheet_url_2: str = "https://docs.google.com/spreadsheets/d/1NfR5zVKSAKAIogEj9ehJppU8beKMTgCigHhCbQ7LXoQ/edit?usp=sharing"
    sheet_url_3: str = "https://docs.google.com/spreadsheets/d/15G4s5UUFf6aGVya4hhh2apLrfl52YLZMMvQ4QzA4qe4/edit?usp=sharing"
    smtp_email: str = "mochammadilhamrizki1@gmail.com"
    smtp_password: str = "uthsaysrvmrpqrku"
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587

class SettingsUpdate(BaseModel):
    store_name: Optional[str] = None
    logo_url: Optional[str] = None
    sheet_url_1: Optional[str] = None
    sheet_url_2: Optional[str] = None
    sheet_url_3: Optional[str] = None

class DashboardStats(BaseModel):
    total_classes: int
    total_subjects: int
    categories: Dict[str, Dict[str, int]]  # {category_id: {classes: n, subjects: n}}

class SheetData(BaseModel):
    classes: List[str]
    subjects: List[str]
    links: Dict[str, Dict[str, str]]  # {class_name: {subject_name: link}}

class UpdateLinkRequest(BaseModel):
    category: str
    class_name: str
    subject_name: str
    link: str

class AddClassRequest(BaseModel):
    category: str
    class_name: str

class AddSubjectRequest(BaseModel):
    category: str
    subject_name: str

class SendEmailRequest(BaseModel):
    category: str
    class_name: str
    subject_name: str
    recipient_email: EmailStr

class SendBulkEmailRequest(BaseModel):
    category: str
    class_names: List[str]  # Multiple classes
    subject_names: List[str]  # Multiple subjects
    recipient_email: EmailStr

class Customer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: EmailStr
    phone: Optional[str] = ""
    notes: Optional[str] = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CustomerCreate(BaseModel):
    name: str
    email: EmailStr
    phone: Optional[str] = ""
    notes: Optional[str] = ""

class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    notes: Optional[str] = None

class EmailLog(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: str
    category_name: str
    recipient_email: str
    recipient_name: Optional[str] = ""
    total_links: int
    details: str  # JSON string with links info
    status: str = "success"
    sent_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== Helper Functions ====================

def create_jwt_token(username: str) -> str:
    """Create JWT token"""
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["username"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_or_create_settings() -> dict:
    """Get or create default settings"""
    settings = await db.settings.find_one()
    if not settings:
        default_settings = Settings().model_dump()
        await db.settings.insert_one(default_settings)
        return default_settings
    return settings

def get_sheet_url_by_category(settings: dict, category: str) -> str:
    """Get sheet URL by category"""
    sheet_key = f"sheet_url_{category}"
    sheet_url = settings.get(sheet_key, "")
    if not sheet_url:
        raise HTTPException(status_code=400, detail=f"Sheet URL untuk kategori {CATEGORIES.get(category, 'Unknown')} belum dikonfigurasi")
    return sheet_url

async def read_google_sheet(sheet_url: str) -> SheetData:
    """Read data from Google Sheets - using public access with CSV export"""
    try:
        # Extract sheet ID from URL
        if "/d/" in sheet_url:
            sheet_id = sheet_url.split("/d/")[1].split("/")[0]
        else:
            raise HTTPException(status_code=400, detail="Invalid sheet URL")
        
        # Use public CSV export endpoint
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid=0"
        
        import requests
        response = requests.get(csv_url)
        response.raise_for_status()
        
        # Parse CSV
        import csv
        from io import StringIO
        
        csv_data = StringIO(response.text)
        reader = csv.reader(csv_data)
        all_values = list(reader)
        
        if len(all_values) < 2:
            return SheetData(classes=[], subjects=[], links={})
        
        # Parse data
        subjects = [cell for cell in all_values[0][1:] if cell]  # Row 1, skip column A
        classes = []
        links = {}
        
        for row in all_values[1:]:
            if row and row[0]:  # If class name exists
                class_name = row[0]
                classes.append(class_name)
                links[class_name] = {}
                
                for i, subject in enumerate(subjects):
                    link = row[i + 1] if i + 1 < len(row) else ""
                    links[class_name][subject] = link
        
        return SheetData(classes=classes, subjects=subjects, links=links)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading sheet: {str(e)}")

async def write_google_sheet(sheet_url: str, data: SheetData):
    """Write data to Google Sheets"""
    # Note: Writing to Google Sheets requires proper authentication
    # For this simplified version, users should edit directly in Google Sheets
    # This function is kept for future implementation with proper credentials
    raise HTTPException(
        status_code=501, 
        detail="Edit langsung di Google Sheets. Aplikasi akan otomatis membaca perubahan Anda."
    )

async def send_email_smtp(recipient: str, subject: str, body: str, settings: dict):
    """Send email via SMTP"""
    try:
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = settings["smtp_email"]
        message["To"] = recipient
        
        html_part = MIMEText(body, "html")
        message.attach(html_part)
        
        await aiosmtplib.send(
            message,
            hostname=settings["smtp_host"],
            port=settings["smtp_port"],
            username=settings["smtp_email"],
            password=settings["smtp_password"],
            start_tls=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

# ==================== Routes ====================

@api_router.get("/categories")
async def get_categories():
    """Get all product categories"""
    return {"categories": [{
        "id": cat_id,
        "name": cat_name
    } for cat_id, cat_name in CATEGORIES.items()]}

@api_router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Admin login"""
    admin = await db.admin.find_one({"username": request.username})
    
    if not admin:
        # Create default admin if not exists
        if request.username == "produkdigital" and request.password == "12345678":
            hashed_pw = bcrypt.hash(request.password)
            await db.admin.insert_one({
                "id": str(uuid.uuid4()),
                "username": "produkdigital",
                "password": hashed_pw
            })
            token = create_jwt_token(request.username)
            return LoginResponse(token=token, username=request.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    if not bcrypt.verify(request.password, admin["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_jwt_token(request.username)
    return LoginResponse(token=token, username=request.username)

@api_router.post("/auth/change-password")
async def change_password(request: ChangePasswordRequest, username: str = Depends(verify_token)):
    """Change admin password"""
    admin = await db.admin.find_one({"username": username})
    
    if not admin:
        raise HTTPException(status_code=404, detail="Admin not found")
    
    # Verify current password
    if not bcrypt.verify(request.current_password, admin["password"]):
        raise HTTPException(status_code=401, detail="Current password incorrect")
    
    # Update password
    new_hashed = bcrypt.hash(request.new_password)
    await db.admin.update_one(
        {"username": username},
        {"$set": {"password": new_hashed}}
    )
    
    return {"message": "Password changed successfully"}

@api_router.get("/settings")
async def get_settings(username: str = Depends(verify_token)):
    """Get settings"""
    settings = await get_or_create_settings()
    # Don't expose sensitive data
    settings.pop("smtp_password", None)
    settings.pop("_id", None)
    return settings

@api_router.put("/settings")
async def update_settings(update: SettingsUpdate, username: str = Depends(verify_token)):
    """Update settings"""
    settings = await get_or_create_settings()
    
    update_data = {k: v for k, v in update.model_dump().items() if v is not None}
    
    if update_data:
        await db.settings.update_one(
            {"id": settings["id"]},
            {"$set": update_data}
        )
    
    return {"message": "Settings updated successfully"}

@api_router.get("/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats(username: str = Depends(verify_token)):
    """Get dashboard statistics"""
    settings = await get_or_create_settings()
    
    total_classes = 0
    total_subjects = 0
    categories_stats = {}
    
    for cat_id in CATEGORIES.keys():
        try:
            sheet_url = get_sheet_url_by_category(settings, cat_id)
            data = await read_google_sheet(sheet_url)
            categories_stats[cat_id] = {
                "classes": len(data.classes),
                "subjects": len(data.subjects)
            }
            total_classes += len(data.classes)
            total_subjects += len(data.subjects)
        except Exception:
            categories_stats[cat_id] = {"classes": 0, "subjects": 0}
    
    return DashboardStats(
        total_classes=total_classes,
        total_subjects=total_subjects,
        categories=categories_stats
    )

@api_router.get("/sheets/data", response_model=SheetData)
async def get_sheet_data(category: str = Query(..., description="Category ID (1, 2, or 3)"), username: str = Depends(verify_token)):
    """Get data from Google Sheets by category"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, category)
    return await read_google_sheet(sheet_url)

@api_router.put("/sheets/update-link")
async def update_link(request: UpdateLinkRequest, username: str = Depends(verify_token)):
    """Update a single link in the sheet"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, request.category)
    
    # Read current data
    data = await read_google_sheet(sheet_url)
    
    # Update link
    if request.class_name in data.links:
        data.links[request.class_name][request.subject_name] = request.link
    
    # Write back
    await write_google_sheet(sheet_url, data)
    
    return {"message": "Link updated successfully"}

@api_router.post("/sheets/add-class")
async def add_class(request: AddClassRequest, username: str = Depends(verify_token)):
    """Add a new class (row)"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, request.category)
    
    # Read current data
    data = await read_google_sheet(sheet_url)
    
    # Add new class
    if request.class_name not in data.classes:
        data.classes.append(request.class_name)
        data.links[request.class_name] = {subject: "" for subject in data.subjects}
    
    # Write back
    await write_google_sheet(sheet_url, data)
    
    return {"message": "Class added successfully"}

@api_router.post("/sheets/add-subject")
async def add_subject(request: AddSubjectRequest, username: str = Depends(verify_token)):
    """Add a new subject (column)"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, request.category)
    
    # Read current data
    data = await read_google_sheet(sheet_url)
    
    # Add new subject
    if request.subject_name not in data.subjects:
        data.subjects.append(request.subject_name)
        for class_name in data.classes:
            data.links[class_name][request.subject_name] = ""
    
    # Write back
    await write_google_sheet(sheet_url, data)
    
    return {"message": "Subject added successfully"}

@api_router.post("/email/send")
async def send_email(request: SendEmailRequest, username: str = Depends(verify_token)):
    """Send email with product link"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, request.category)
    
    # Get link from sheet
    data = await read_google_sheet(sheet_url)
    
    if request.class_name not in data.links:
        raise HTTPException(status_code=404, detail="Class not found")
    
    if request.subject_name not in data.links[request.class_name]:
        raise HTTPException(status_code=404, detail="Subject not found")
    
    link = data.links[request.class_name][request.subject_name]
    
    if not link:
        raise HTTPException(status_code=400, detail="No link available for this class and subject")
    
    # Prepare email
    store_name = settings.get("store_name", "Pena Digital Store")
    category_name = CATEGORIES.get(request.category, "Produk Digital")
    subject = f"Link Program Digital - {category_name} - {request.class_name} - {request.subject_name}"
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #FF6B35;">{store_name}</h2>
            <p>Terima kasih telah membeli program digital kami!</p>
            
            <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
                <p><strong>Kategori:</strong> {category_name}</p>
                <p><strong>Kelas:</strong> {request.class_name}</p>
                <p><strong>Mata Pelajaran:</strong> {request.subject_name}</p>
                <p><strong>Link Produk Digital:</strong></p>
                <a href="{link}" style="color: #FF6B35; word-break: break-all;">{link}</a>
            </div>
            
            <p>Silakan klik link di atas untuk mengakses program digital Anda.</p>
            <p>Jika ada pertanyaan, jangan ragu untuk menghubungi kami.</p>
            
            <p style="margin-top: 30px; color: #666; font-size: 14px;">Salam,<br>{store_name}</p>
        </div>
    </body>
    </html>
    """
    
    # Send email
    await send_email_smtp(request.recipient_email, subject, body, settings)
    
    # Log activity
    log_entry = EmailLog(
        category=request.category,
        category_name=CATEGORIES.get(request.category, "Unknown"),
        recipient_email=request.recipient_email,
        total_links=1,
        details=f'{{"class": "{request.class_name}", "subject": "{request.subject_name}", "link": "{link}"}}',
        status="success"
    )
    log_doc = log_entry.model_dump()
    log_doc['sent_at'] = log_doc['sent_at'].isoformat()
    await db.email_logs.insert_one(log_doc)
    
    return {"message": "Email sent successfully"}

# ==================== Customer Management ====================

@api_router.get("/customers", response_model=List[Customer])
async def get_customers(username: str = Depends(verify_token)):
    """Get all customers"""
    customers = await db.customers.find({}, {"_id": 0}).to_list(1000)
    for customer in customers:
        if isinstance(customer.get('created_at'), str):
            customer['created_at'] = datetime.fromisoformat(customer['created_at'])
    return customers

@api_router.post("/customers", response_model=Customer)
async def create_customer(customer: CustomerCreate, username: str = Depends(verify_token)):
    """Create new customer"""
    # Check if email already exists
    existing = await db.customers.find_one({"email": customer.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already exists")
    
    customer_obj = Customer(**customer.model_dump())
    doc = customer_obj.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    
    await db.customers.insert_one(doc)
    return customer_obj

@api_router.put("/customers/{customer_id}")
async def update_customer(customer_id: str, update: CustomerUpdate, username: str = Depends(verify_token)):
    """Update customer"""
    customer = await db.customers.find_one({"id": customer_id})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    update_data = {k: v for k, v in update.model_dump().items() if v is not None}
    
    if update_data:
        await db.customers.update_one(
            {"id": customer_id},
            {"$set": update_data}
        )
    
    return {"message": "Customer updated successfully"}

@api_router.delete("/customers/{customer_id}")
async def delete_customer(customer_id: str, username: str = Depends(verify_token)):
    """Delete customer"""
    result = await db.customers.delete_one({"id": customer_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    return {"message": "Customer deleted successfully"}

# ==================== Email Logs ====================

@api_router.get("/logs", response_model=List[EmailLog])
async def get_email_logs(
    limit: int = 100,
    skip: int = 0,
    category: Optional[str] = None,
    username: str = Depends(verify_token)
):
    """Get email logs with pagination and filters"""
    query = {}
    if category:
        query["category"] = category
    
    logs = await db.email_logs.find(query, {"_id": 0}).sort("sent_at", -1).skip(skip).limit(limit).to_list(limit)
    
    for log in logs:
        if isinstance(log.get('sent_at'), str):
            log['sent_at'] = datetime.fromisoformat(log['sent_at'])
    
    return logs

@api_router.delete("/logs")
async def clear_logs(username: str = Depends(verify_token)):
    """Clear all email logs"""
    await db.email_logs.delete_many({})
    return {"message": "All logs cleared successfully"}

# ==================== Email Sending ====================

@api_router.post("/email/send-bulk")
async def send_bulk_email(request: SendBulkEmailRequest, username: str = Depends(verify_token)):
    """Send email with multiple product links"""
    settings = await get_or_create_settings()
    sheet_url = get_sheet_url_by_category(settings, request.category)
    
    # Get data from sheet
    data = await read_google_sheet(sheet_url)
    
    # Collect all links
    links_data = []
    for class_name in request.class_names:
        if class_name not in data.links:
            continue
        
        for subject_name in request.subject_names:
            if subject_name not in data.links[class_name]:
                continue
            
            link = data.links[class_name][subject_name]
            if link:
                links_data.append({
                    "class": class_name,
                    "subject": subject_name,
                    "link": link
                })
    
    if not links_data:
        raise HTTPException(status_code=400, detail="No links available for selected classes and subjects")
    
    # Prepare email
    store_name = settings.get("store_name", "Pena Digital Store")
    category_name = CATEGORIES.get(request.category, "Produk Digital")
    
    # Build subject
    if len(request.class_names) == 1 and len(request.subject_names) > 1:
        subject = f"Link Program Digital - {category_name} - {request.class_names[0]} - Paket {len(links_data)} Mapel"
    elif len(request.class_names) > 1 and len(request.subject_names) == 1:
        subject = f"Link Program Digital - {category_name} - {request.subject_names[0]} - {len(links_data)} Kelas"
    else:
        subject = f"Link Program Digital - {category_name} - Paket {len(links_data)} Produk"
    
    # Build links HTML
    links_html = ""
    for idx, item in enumerate(links_data, 1):
        links_html += f"""
        <div style="background-color: #f0f0f0; padding: 12px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #FF6B35;">
            <p style="margin: 0; font-weight: bold; color: #333;">{idx}. {item['class']} - {item['subject']}</p>
            <a href="{item['link']}" style="color: #FF6B35; word-break: break-all; text-decoration: none; display: block; margin-top: 5px;">{item['link']}</a>
        </div>
        """
    
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
            <h2 style="color: #FF6B35;">{store_name}</h2>
            <p>Terima kasih telah membeli program digital kami!</p>
            
            <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 20px 0; border: 2px solid #FF6B35;">
                <p style="margin: 0;"><strong>Kategori:</strong> {category_name}</p>
                <p style="margin: 5px 0 0 0;"><strong>Total Produk:</strong> {len(links_data)} link</p>
            </div>
            
            <h3 style="color: #FF6B35; border-bottom: 2px solid #FF6B35; padding-bottom: 10px;">Link Produk Digital Anda:</h3>
            
            {links_html}
            
            <p style="margin-top: 20px;">Silakan klik link di atas untuk mengakses program digital Anda.</p>
            <p>Jika ada pertanyaan, jangan ragu untuk menghubungi kami.</p>
            
            <p style="margin-top: 30px; color: #666; font-size: 14px;">Salam,<br>{store_name}</p>
        </div>
    </body>
    </html>
    """
    
    # Send email
    await send_email_smtp(request.recipient_email, subject, body, settings)
    
    # Log activity
    import json
    log_entry = EmailLog(
        category=request.category,
        category_name=CATEGORIES.get(request.category, "Unknown"),
        recipient_email=request.recipient_email,
        total_links=len(links_data),
        details=json.dumps([{
            "class": item["class"],
            "subject": item["subject"],
            "link": item["link"]
        } for item in links_data]),
        status="success"
    )
    log_doc = log_entry.model_dump()
    log_doc['sent_at'] = log_doc['sent_at'].isoformat()
    await db.email_logs.insert_one(log_doc)
    
    return {
        "message": "Email sent successfully",
        "total_links": len(links_data)
    }

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
