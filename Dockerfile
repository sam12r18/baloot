# استفاده از پایتون به عنوان پایه
FROM python:3.11-slim

# تنظیم دایرکتوری کاری داخل کانتینر
WORKDIR /app

# کپی کردن فایل‌های پروژه به کانتینر
COPY . /app

# نصب ابزارهای مورد نیاز
RUN apt-get update && apt-get install -y \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ساخت محیط‌های مجازی و نصب بسته‌ها
RUN python -m venv /venv_transparent && \
    /venv_transparent/bin/pip install --upgrade pip && \
    /venv_transparent/bin/pip install --timeout=120 ./app/background_remove/transparent-background-git/ && \
    \
    python -m venv /venv_upscale && \
    /venv_upscale/bin/pip install --upgrade pip && \
    /venv_upscale/bin/pip install ./app/background_remove/image-upscale/ && \
    \
    python -m venv /venv_faceorienter && \
    /venv_faceorienter/bin/pip install --upgrade pip && \
    /venv_faceorienter/bin/pip install ./app/background_remove/faceorienter/ && \
    \
    python -m venv /venv_pypi && \
    /venv_pypi/bin/pip install --upgrade pip && \
    /venv_pypi/bin/pip install retina-face

# باز کردن پورت برای دسترسی به اپلیکیشن
EXPOSE 8000

# دستور اجرای پروژه
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
