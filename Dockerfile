FROM python:3.9

# إنشاء مستخدم عادي للأمان
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# نسخ ملف الـ requirements من جوه فولدر backend وتصطيب المكتبات
COPY --chown=user ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# نسخ فولدر الـ artifacts والـ backend بالكامل جوه الـ container
COPY --chown=user ./artifacts /app/artifacts
COPY --chown=user ./backend /app

# تشغيل السيرفر من المسار المظبوط
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]