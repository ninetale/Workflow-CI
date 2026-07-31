# Menggunakan base image Python yang ringan
FROM python:3.10-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file dependensi
COPY requirements.txt .

# Menginstal library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh file proyek ke dalam container
COPY . .

# Menjalankan aplikasi FastAPI (menggunakan inference.py)
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]