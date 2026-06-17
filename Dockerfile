FROM python:3.9

# إنشاء مستخدم عادي للأمان
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# نسخ ملف الـ requirements وتصطيب المكتبات
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# نسخ كل ملفات الكود من فولدر backend إلى داخل الـ container
COPY --chown=user . /app

# نسخ فولدر الـ artifacts اللي بره وجعله في نفس مستوى الكود جوه الـ container
COPY --chown=user ../artifacts /app/artifacts

# تشغيل السيرفر (شيلنا البورت الإجباري عشان Railway يربطه تلقائي)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]