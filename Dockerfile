FROM python:3.9

# إنشاء مستخدم عادي للأمان
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# نسخ ملف الـ requirements من مكانه المباشر جوه الفولدر
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# نسخ كل الملفات اللي جوه الـ backend للـ container
COPY --chown=user . /app

# تشغيل السيرفر
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]