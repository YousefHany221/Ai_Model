FROM python:3.9

# إنشاء مستخدم عادي للأمان
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# نسخ ملف الـ requirements وتصطيب المكتبات (الملف جنبه علطول دلوقتي)
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# نسخ كل ملفات الكود (main.py, db.py, إلخ) اللي جنبه جوه الـ container
COPY --chown=user . /app

# تشغيل السيرفر من الـ app علطول
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]