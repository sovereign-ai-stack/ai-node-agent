# ⚡ عامل هوشمند نود پردازش گرافیکی (Enterprise AI Node Agent)
## راهنمای جامع استقرار در محیط عملیاتی (Production Deployment Guide)

این پکیج یک عامل سبک و خودکار (Standalone Agent) است که روی سرورها و رایانه‌های مجهز به کارت گرافیک انویدیا اجرا شده و قدرت پردازش آن‌ها را از طریق شبکه به **پلتفرم مرکزی هوش مصنوعی (Central Control Plane)** پیوند می‌زند.

---

### ۱. قابلیت‌های کلیدی در پروداکشن

* **کشف خودکار سخت‌افزار (Auto Hardware Detection):** اندازه‌گیری خودکار VRAM کارت گرافیک، نسل پردازنده و انتخاب بهترین معماری کوانتیزاسیون.
* **ثبت خودکار در رجیستری مرکزی:** ارسال مدل و متادیتا به رجیستری مرکزی (`:8200`) و تزریق خودکار به مسیرهای پروکسی LiteLLM در کمتر از ۱ ثانیه.
* **ضربان قلب پیوسته (Heartbeat):** ارسال گزارش سلامت و حافظه هر ۱۵ ثانیه برای مدیریت توزیع بار و Failover.
* **استقرار کاملاً آفلاین (Air-Gapped Ready):** امکان لود وزن‌های مدل از دیسک محلی و ایمیج آفلاین بدون نیاز به اینترنت.

---

### ۲. پیش‌نیازهای سرور یا سیستم گرافیکی

1. **درایور انویدیا (NVIDIA Driver):** نسخه 535 یا بالاتر با پشتیبانی از CUDA 12.1+
   ```bash
   nvidia-smi
   ```
2. **داکر و تولکیت کانتینر انویدیا (NVIDIA Container Toolkit):**
   ```bash
   # بررسی فعال بودن انویدیا در داکر
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
3. **پایتون:** نسخه 3.10 یا بالاتر

---

### ۳. شبکه‌سازی امن کلاستر با Tailscale (حیاتی)

برای اینکه این نود بتواند از پشت فایروال، NAT یا اینترنت خانگی/سازمانی بدون نیاز به آی‌پی پابلیک به سرور مرکزی متصل شود:

#### گام ۱: نصب و ورود به شبکه Tailscale
```bash
# نصب روی سرور لینوکس نود
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# یا در ویندوز: نصب نرم‌افزار Tailscale از وبسایت رسمی و لاگین با همان اکانت سرور مرکزی
```

#### گام ۲: تست دسترسی به سرور مرکزی
آدرس IP سرور مرکزی در شبکه Tailscale (مثلاً `100.115.80.12`) را پینگ کنید:
```bash
curl -I http://100.115.80.12:8200/health
# باید پاسخ HTTP/1.1 200 OK دریافت کنید.
```

---

### ۴. پیکربندی متغیرهای محیطی نود (`.env`)

فایل `.env.example` را به `.env` کپی کرده و مقادیر را بر اساس سیستم خود پر کنید:

```ini
# ۱. آدرس رجیستری سرور مرکزی (در شبکه Tailscale یا محلی)
GATEWAY_URL=http://100.115.80.12:8200

# ۲. شناسه اختصاصی و خوانای این نود
NODE_ID=worker-rtx4090-node1

# ۳. پورت سرویس‌دهی موتور vLLM روی این سیستم
VLLM_PORT=8000

# ۴. مسیر ذخیره مدل‌های دانلود شده روی هارد دیسک (برای عملکرد آفلاین)
LOCAL_MODEL_PATH=/data/models/Qwen2.5-Coder-7B-Instruct

# ۵. حالت غیرتعاملی (بدون سوال و جواب - مناسب سرورهای پروداکشن)
NON_INTERACTIVE=true
```

---

### ۵. روش‌های استقرار و اجرا

#### روش ۱: اجرای مستقیم با خط فرمان (CLI)
```bash
cd ai-node-agent
pip install -r requirements.txt

# اجرای نود و اتصال خودکار به سرور مرکزی
python node_agent.py --gateway-url http://100.115.80.12:8200
```

#### روش ۲: اجرای خودکار به عنوان سرویس سیستمی دائم (Linux systemd)
برای اینکه با ریستارت شدن سرور، نود ایجنت به صورت خودکار بالا بیاید:

یک فایل سرویس در `/etc/systemd/system/ai-node-agent.service` بسازید:
```ini
[Unit]
Description=Enterprise AI Node Agent Daemon
After=network.target tailscaled.service docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ai-node-agent
ExecStart=/usr/bin/python3 /opt/ai-node-agent/node_agent.py --gateway-url http://100.115.80.12:8200 --non-interactive
Restart=always
RestartSec=10
EnvironmentFile=/opt/ai-node-agent/.env

[Install]
WantedBy=multi-user.target
```

فعال‌سازی و شروع سرویس:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-node-agent
sudo systemctl start ai-node-agent

# مشاهده لاگ‌های زنده نود
sudo journalctl -u ai-node-agent -f
```

#### روش ۳: اجرای سریع در ویندوز
روی فایل `run.bat` دابل‌کلیک کنید یا در PowerShell اجرا کنید:
```powershell
.\run.bat
```

---

### ۶. پایش و عیب‌یابی نود در محیط واقعی

* **بررسی مصرف حافظه گرافیکی و دما:**
  ```bash
  watch -n 1 nvidia-smi
  ```
* **بررسی کانتینر vLLM در حال اجرا:**
  ```bash
  docker ps | grep vllm
  docker logs -f vllm-worker
  ```
* **بررسی ثبت نود در سرور مرکزی:**
  با مراجعه به پنل وب ادمین در سرور مرکزی (`http://<CENTRAL_IP>:8400/admin`)، نود خود را در تب نودهای فعال مشاهده خواهید کرد.
