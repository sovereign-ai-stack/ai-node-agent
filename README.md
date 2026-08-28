# 🚀 راهنمای راه‌اندازی نود محلی هوش مصنوعی (Enterprise AI Node Agent)

این پکیج به صورت کاملاً مستقل (Standalone) طراحی شده است تا روی سرورهای محلی سازمان‌ها و شعبات اجرا شده و به صورت خودکار به پلتفرم مرکزی متصل شود.

---

## 📋 پیش‌نیازهای سرور محلی
1. **نصب درایور انویدیا (NVIDIA Drivers):** حداقل نسخه CUDA 12.1+
2. **داکر و ابزار انویدیا (Docker + NVIDIA Container Toolkit):**
   ```bash
   # تست صحت نصب درایور و داکر
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
3. **پایتون:** نسخه 3.8 یا بالاتر

---

## ⚡ نحوه اجرا و اتصال به پلتفرم مرکزی (تنها با یک دستور)

### ۱. نصب وابستگی‌های بسیار سبک (اختیاری):
```bash
pip install -r requirements.txt
```

### ۲. اجرای نود و ثبت خودکار در گیتوی مرکزی:
```bash
python node_agent.py --gateway-url http://<CENTRAL_SERVER_IP>:8200
```
*(به جای `<CENTRAL_SERVER_IP>`، آدرس IP یا دامنه سرور مرکزی سازمان را وارد کنید)*

---

## 🔒 استقرار در سرورهای کاملاً آفلاین (Air-Gapped)
اگر سرور سازمان دسترسی به اینترنت ندارد و مدل‌ها قبلاً روی دیسک دانلود شده‌اند:
```bash
python node_agent.py \
  --gateway-url http://<CENTRAL_SERVER_IP>:8200 \
  --local-model-path /data/models/Qwen2.5-7B-Instruct-AWQ \
  --served-name qwen-7b
```

---

## 🛠️ گزینه‌های خط فرمان (CLI Options)

| سوئیچ | توضیحات | پیش‌فرض |
|---|---|---|
| `--gateway-url` | آدرس رجیستری سرور مرکزی جهت ثبت خودکار و ارسال هارت‌بیت | اختیاری (Standalone) |
| `--port` | پورت لوکال برای سرویس‌دهی vLLM | `8000` |
| `--local-model-path` | مسیر لوکال وزن‌های مدل روی دیسک (برای محیط‌های ایزوله) | `None` |
| `--tier` | انتخاب دستی سطح مدل (مثلاً `tier-8gb-vram`, `tier-24gb-vram`) | تشخیص خودکار |
| `--export-compose` | ذخیره تنظیمات بهینه در یک فایل `docker-compose.yml` بدون اجرا | `None` |
| `--probe-only` | فقط بررسی سخت‌افزار و نمایش گزارش تشخیصی | `False` |
| `--heartbeat-interval` | فاصله زمانی ارسال پینگ ضربان قلب به سرور مرکزی (ثانیه) | `15` |
