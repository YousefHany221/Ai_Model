FROM python:3.9

# إنشاء مستخدم عادي للأمان (Hugging Face بيطلب ده)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# نسخ ملف الـ requirements وتصطب المكتبات
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# نسخ كل ملفات المشروع والـ artifacts جوه الـ container
COPY --chown=user . /app

# تشغيل السيرفر على بورت 7860 الإجباري لـ Hugging Face
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]