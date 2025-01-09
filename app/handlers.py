import json
from fastapi import UploadFile, HTTPException  # اضافه کردن HTTPException برای مدیریت خطاها

from image_processor import ImageProcessor
from PIL import Image

async def upload_image_handler(file: UploadFile, setting: str, user_id: int):
    try:
        # تبدیل رشته JSON به دیکشنری
        setting = json.loads(setting)
#تنظیمات باید یک فایل جیسون با این ساختار باشد
#{
#"upscale": true,
#  "fix_orientation": true,
#  "rotate_and_resize_face": {
#    "enabled": true,
#    "face_height": 450,
#    "distance_top": 10,
#    "eye_to_eye": 1,
#    "eye_to_bottom": 1,
#    "eye_to_chin": 20
#  },
#  "resize_and_shift_image": {
#    "enabled": true,
#    "target_width": 50,
#    "target_height": 70,
#    "shift_x_pixels": 10,
#    "shift_y_pixels": 5
#  },
#  "bg_color": "green",
#  "change_brightness": true,
#  "brightness_alpha": 1.2,
#  "brightness_beta": 15
#}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="تنظیمات ارسال شده صحیح نیست.")

    # ذخیره تصویر در مسیر static
    file_path = f"static/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # ایجاد شی از کلاس ImageProcessor
    processor = ImageProcessor()

    # حذف پس‌زمینه تصویر
    output_path = processor.remove_background(file_path)

    # افزایش کیفیت تصویر (اگر تنظیم شده باشد)
    if setting.get("upscale", False):
        output_path = processor.upscale_image(file_path)

    # اصلاح جهت تصویر (اگر تنظیم شده باشد)
    if setting.get("fix_orientation", False):
        output_path = processor.fix_orientation(output_path)

    # چرخش و تغییر اندازه چهره (با تنظیمات پیش‌فرض)
    #output_path = processor.rotate_and_resize_face(output_path, face_height=400, distance_top=5, eye_to_eye=0, eye_to_bottom=0, eye_to_chin=18, metric="mm")
    # چرخش و تغییر اندازه چهره (اگر تنظیم شده باشد)
    rotate_resize = setting.get("rotate_and_resize_face", {})
    if rotate_resize.get("enabled", False):  # بررسی اینکه چرخش و تغییر اندازه فعال باشد
        output_path = processor.rotate_and_resize_face(
            output_path,
            face_height=rotate_resize.get("face_height", 400),
            distance_top=rotate_resize.get("distance_top", 5),
            eye_to_eye=rotate_resize.get("eye_to_eye", 0),
            eye_to_bottom=rotate_resize.get("eye_to_bottom", 0),
            eye_to_chin=rotate_resize.get("eye_to_chin", 18),
            metric="mm"
        )
        
    # تغییر اندازه و جابجایی تصویر (با تنظیمات پیش‌فرض)
    #output_path = processor.resize_and_shift_image(output_path, target_width=40, target_height=60, top_cut=0, shift_x_pixels=0, shift_y_pixels=0, metric="mm")
    # تغییر اندازه و جابجایی تصویر (اگر تنظیم شده باشد)
    resize_shift = setting.get("resize_and_shift_image", {})
    if resize_shift.get("enabled", False):  # بررسی اینکه تغییر اندازه و جابجایی فعال باشد
        output_path = processor.resize_and_shift_image(
            output_path,
            target_width=resize_shift.get("target_width", 40),
            target_height=resize_shift.get("target_height", 60),
            top_cut=resize_shift.get("top_cut", 0),
            shift_x_pixels=resize_shift.get("shift_x_pixels", 0),
            shift_y_pixels=resize_shift.get("shift_y_pixels", 0),
            metric="mm"
        )
        
    # تنظیم رنگ پس‌زمینه (در صورتی که نیاز باشد)
    bg_color = setting.get('bg_color', 'transparent')  # استفاده از پیش‌فرض 'transparent'
    output_path = processor.bg_color(output_path, bg_color)

    # تغییر روشنایی تصویر (اگر تنظیم شده باشد)
    if setting.get("change_brightness", False):
        output_path = processor.change_brightness(output_path, alpha=setting.get("brightness_alpha", 1.0), beta=setting.get("brightness_beta", 10))

    # برگرداندن مسیر فایل پردازش‌شده
    return {"filename": file.filename, "output_path": output_path}

def get_image_handler(filename: str):
    # بازگرداندن مسیر فایل برای دریافت تصویر
    return {"image_path": f"static/{filename}"}
