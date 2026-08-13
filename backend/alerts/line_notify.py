# alerts/line_notify.py — ZENTRA LINE OA Alert Engine
# Slide: ส่งผ่าน LINE OA 3 ระดับ (warning/alert/emergency)
# Features: cooldown, retry, background queue, daily report
# ================================================================

from __future__ import annotations
import time
import threading
import requests
import cv2
from datetime import datetime
from typing import Optional


def _cfg():
    import config as c
    return c


# ── Cooldown (thread-safe) ──────────────────────────────────
_lock           = threading.Lock()
_last_sent:     dict[str, float] = {}
# Per-group throttle for the "all groups" routing model. Keyed by "<group>|<level>"
# so a warning sent to a group can NEVER suppress a later emergency to that same
# group — only repeats of the same level+group are throttled.
_last_group_sent: dict[str, float] = {}
_alert_queue:   list[dict]       = []
_queue_lock     = threading.Lock()
_running        = False
_sender_thread: Optional[threading.Thread] = None

MAX_RETRIES     = 3
RETRY_DELAY_SEC = 2.0


# ================================================================
# IMAGE UPLOAD
# ================================================================
def upload_image(image, timeout: int = 15) -> str:
    """อัพโหลดภาพไป catbox.moe → URL"""
    try:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return ""
        resp = requests.post(
            _cfg().IMAGE_UPLOAD_URL,
            data={"reqtype": "fileupload"},
            files={"fileToUpload": ("zentra.jpg", buf.tobytes(), "image/jpeg")},
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.text.startswith("https://"):
            url = resp.text.strip()
            print(f"[LINE] Image uploaded: {url}")
            return url
    except requests.exceptions.Timeout:
        print("[LINE] Image upload timeout")
    except Exception as e:
        print(f"[LINE] Image upload error: {e}")
    return ""


# ================================================================
# CORE SEND (with retry)
# ================================================================
def _send_to_group(group_id: str, msg: str, img_url: str = "") -> bool:
    cfg   = _cfg()
    token = cfg.LINE_OA_CHANNEL_ACCESS_TOKEN
    if not token or not group_id:
        return False

    headers  = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    messages: list[dict] = [{"type": "text", "text": msg}]
    if img_url:
        messages.append({
            "type": "image",
            "originalContentUrl": img_url,
            "previewImageUrl":    img_url,
        })

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.line.me/v2/bot/message/push",
                headers=headers,
                json={"to": group_id, "messages": messages},
                timeout=12,
            )
            if resp.status_code == 200:
                return True
            # Rate limit → รอแล้ว retry
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", "5"))
                print(f"[LINE] Rate limited, wait {retry_after}s...")
                time.sleep(retry_after)
                continue
            print(f"[LINE] Send failed ({resp.status_code}): {resp.text[:100]}")
            return False
        except requests.exceptions.Timeout:
            print(f"[LINE] Timeout attempt {attempt}/{MAX_RETRIES}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SEC)
        except Exception as e:
            print(f"[LINE] Error: {e}")
            return False
    return False


# ================================================================
# PUBLIC API
# ================================================================
def send_line_notify(
    msg:          str,
    image=None,
    level:        str  = "warning",
    cooldown_key: str  = "default",
    cooldown_sec: Optional[int] = None,
    async_send:   bool = True,
) -> bool:
    cfg = _cfg()

    # PDPA: optionally keep person images on-device (text-only LINE alert)
    if not getattr(cfg, "LINE_UPLOAD_IMAGES", True):
        image = None

    # เลือก cooldown ตาม level
    if cooldown_sec is None:
        if level == cfg.ALERT_LEVEL_EMERGENCY:
            cooldown_sec = cfg.FALL_COOLDOWN_SECONDS
        elif level == cfg.ALERT_LEVEL_ALERT:
            cooldown_sec = cfg.ZONE_COOLDOWN_SECONDS
        else:
            cooldown_sec = cfg.VIOLATION_COOLDOWN_SECONDS

    # cooldown check
    now = time.time()
    with _lock:
        if now - _last_sent.get(cooldown_key, 0.0) < cooldown_sec:
            remaining = int(cooldown_sec - (now - _last_sent[cooldown_key]))
            print(f"[LINE] Cooldown '{cooldown_key}': {remaining}s remaining")
            return False
        _last_sent[cooldown_key] = now

    # สร้าง message
    ts       = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    icon_map = {
        cfg.ALERT_LEVEL_WARNING:   "⚠️",
        cfg.ALERT_LEVEL_ALERT:     "🔴",
        cfg.ALERT_LEVEL_EMERGENCY: "🆘",
    }
    icon     = icon_map.get(level, "ℹ️")
    full_msg = f"{icon} [ZENTRA {level.upper()}]\n{msg}\n📅 {ts}"

    recipients = [r for r in cfg.ALERT_RECIPIENTS.get(level, [cfg.LINE_OA_GROUP_SUPERVISOR]) if r]
    if not recipients:
        print(f"[LINE] No recipients for level='{level}'")
        return False

    payload = {
        "msg":        full_msg,
        "image":      image.copy() if image is not None else None,
        "recipients": recipients,
        "level":      level,
    }

    if async_send:
        # Only queue when the background sender loop is actually running (the legacy
        # CLI main.py starts it via start_sender()). The FastAPI app path never
        # starts it, so appending here would drop the alert into a queue nobody
        # drains — a silent lost alert. Fall back to a synchronous dispatch there so
        # an alert is never lost regardless of who the caller is.
        if _running:
            with _queue_lock:
                _alert_queue.append(payload)
            return True
        return _dispatch(payload)
    return _dispatch(payload)


def _dispatch(payload: dict) -> bool:
    cfg     = _cfg()
    cd_map  = getattr(cfg, "LINE_GROUP_COOLDOWN", {}) or {}
    level   = payload.get("level", "")
    now     = time.time()
    img_url = upload_image(payload["image"]) if payload.get("image") is not None else ""
    ok = True
    for gid in payload["recipients"]:
        # Per-group throttle (Settings → หน่วงเวลา). Keyed by group+level so distinct
        # alert types to the same group don't cancel each other out.
        cd = int(cd_map.get(gid, 0) or 0)
        if cd > 0:
            key = f"{gid}|{level}"
            with _lock:
                if now - _last_group_sent.get(key, 0.0) < cd:
                    print(f"[LINE] Group cooldown '{gid[:12]}' ({level}): skip")
                    continue
                _last_group_sent[key] = now
        result = _send_to_group(gid, payload["msg"], img_url)
        if result:
            print(f"[LINE] ✅ Sent → {gid[:12]}...")
        ok = ok and result
    return ok


# ================================================================
# BACKGROUND SENDER THREAD
# ================================================================
def start_sender():
    global _sender_thread, _running
    _running = True
    _sender_thread = threading.Thread(target=_sender_loop, daemon=True, name="LINE-Sender")
    _sender_thread.start()
    print("[LINE] Background sender started ✅")


def stop_sender():
    global _running
    _running = False
    # flush queue ที่เหลือ
    remaining = 0
    with _queue_lock:
        remaining = len(_alert_queue)
    if remaining:
        print(f"[LINE] Flushing {remaining} queued alerts...")
        time.sleep(1.5)


def _sender_loop():
    while _running:
        payload = None
        with _queue_lock:
            if _alert_queue:
                payload = _alert_queue.pop(0)
        if payload:
            _dispatch(payload)
        else:
            time.sleep(0.15)


# ================================================================
# DAILY SAFETY REPORT — Slide: ส่งทุกวัน 20:00 น.
# ================================================================
def send_daily_report(stats: dict, report_image=None) -> bool:
    """Push the daily summary to every enabled LINE group.

    Returns True only if at least one group actually received it. It used to
    return None unconditionally — with no token or no group id the recipient list
    is empty, the loop ran zero times, and the caller (the History "LINE" button)
    reported SUCCESS for a message that was never sent.
    """
    ts  = datetime.now().strftime("%d/%m/%Y")
    msg = (
        f"📊 ZENTRA รายงานความปลอดภัยประจำวัน\n"
        f"📅 วันที่: {ts}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🪖 การตรวจจับ PPE : {stats.get('ppe_violations', 0)} ครั้ง\n"
        f"⛔ การตรวจจับพื้นที่ : {stats.get('zone_intrusions', 0)} ครั้ง\n"
        f"🆘 การตรวจจับการล้ม : {stats.get('fall_events', 0)} ครั้ง\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ ระบบ ZENTRA ทำงานปกติ"
    )
    cfg = _cfg()
    if not cfg.LINE_OA_CHANNEL_ACCESS_TOKEN:
        print("[LINE] Daily report NOT sent: no channel access token")
        return False
    # All enabled groups get the daily report (same "all groups" model as alerts).
    recipients = [g for g in dict.fromkeys(getattr(cfg, "LINE_ALL_GROUPS", []) or []) if g]
    if not recipients:
        recipients = list({cfg.LINE_OA_GROUP_SUPERVISOR, cfg.LINE_OA_GROUP_SAFETY} - {""})
    if not recipients:
        print("[LINE] Daily report NOT sent: no group id configured")
        return False
    img_url = upload_image(report_image) if report_image is not None else ""
    ok = False
    for gid in recipients:
        if _send_to_group(gid, msg, img_url):
            ok = True
    print(f"[LINE] Daily report → {len(recipients)} group(s), delivered={ok}")
    return ok


send_line_message = send_line_notify
