# app/route.py

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from handlers import *  # فرض می‌کنیم که اینجا منطق پردازش تصویر وجود دارد

router = APIRouter()

@router.post("/upload/")
async def upload_image(
    file: UploadFile = File(...),
    setting: str = Form(...),  # پارامتر اضافی به نام description
    user_id: int = Form(...),  # پارامتر اضافی به نام user_id
):
    # بررسی اینکه فایل یک تصویر است
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="فایل آپلود شده باید یک تصویر باشد.")
    
    try:
        # ارسال فایل و پارامترهای اضافی به تابع پردازش تصویر
        file_path = await upload_image_handler(file, setting, user_id)
        return {"message": "تصویر با موفقیت پردازش شد", "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"آپلود تصویر با خطا مواجه شد: {str(e)}")

@router.get("/image/{filename}")
def get_image(filename: str):
    try:
        image_path = get_image_handler(filename)  # فرض می‌کنیم که این تابع مسیر کامل تصویر را برمی‌گرداند
        return FileResponse(image_path)  # ارسال تصویر به عنوان پاسخ
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="تصویر یافت نشد.")
