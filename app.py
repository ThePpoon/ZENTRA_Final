import os
import sys
import threading
import time
import webview
from pathlib import Path

# The ZENTRA AI modules print emoji / Thai (✅ ⚠️ 🪖 →) at import and at
# runtime. On a Windows cp1252 console these prints raise
# UnicodeEncodeError, which would crash pipeline startup. Force UTF-8 with
# errors='replace' so logging can never take the app down.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def start_server():
    import uvicorn
    from server.api import app as fastapi_app
    uvicorn.run(
        fastapi_app,
        host="127.0.0.1", #192.168.1.64
        port=7788,
        log_level="warning",
        # Suppress uvicorn shutdown errors on Windows
        timeout_graceful_shutdown=2,
    )


class JsApi:
    # All native dialogs go through pywebview's own create_file_dialog — NOT
    # tkinter. js_api handlers run on a background worker thread, and on macOS a
    # tkinter Tk() created off the main thread deadlocks/aborts (Cocoa UI must
    # live on the main thread). pywebview dispatches its dialog to the GUI thread
    # itself, so it works from here on every platform.

    @staticmethod
    def _window():
        return webview.windows[0] if webview.windows else None

    def open_file_dialog(self):
        w = self._window()
        if not w:
            return ""
        result = w.create_file_dialog(
            webview.OPEN_DIALOG,
            allow_multiple=False,
            file_types=(
                "Video files (*.mp4;*.avi;*.mov;*.mkv;*.m4v;*.wmv)",
                "All files (*.*)",
            ),
        )
        if not result:
            return ""
        return result[0] if isinstance(result, (list, tuple)) else str(result)

    def _ask_save(self, filename: str) -> str:
        """Native Save dialog → chosen path (or '' if cancelled)."""
        w = self._window()
        if not w:
            return ""
        result = w.create_file_dialog(webview.SAVE_DIALOG, save_filename=filename)
        if not result:
            return ""
        return result[0] if isinstance(result, (list, tuple)) else str(result)

    def save_file(self, content: str, filename: str):
        path = self._ask_save(filename)
        if path:
            try:
                Path(path).write_text(content, encoding="utf-8-sig")
            except Exception as e:
                print(f"[JsApi] save_file error: {e}")
                return ""
        return path

    def save_binary(self, b64_content: str, filename: str):
        """Open a native Save dialog and write base64-decoded bytes (PDF, etc.).
        The WebView does not reliably trigger blob downloads, so binary files are
        saved through this bridge instead."""
        import base64
        path = self._ask_save(filename)
        if path:
            try:
                Path(path).write_bytes(base64.b64decode(b64_content))
            except Exception as e:
                print(f"[JsApi] save_binary error: {e}")
                return ""
        return path

    def toggle_fullscreen(self):
        if webview.windows:
            webview.windows[0].toggle_fullscreen()

    def close_app(self):
        """Quit from an in-app button. destroy() must NOT be called directly here:
        this handler runs on the js_api worker thread, and destroy() then tears
        down the very webview that thread is bridged to → the call never returns
        and the window looks frozen. Defer it to a fresh thread so this handler
        returns first; destroy() then fires the 'closing' event → background
        shutdown, exactly like the native close."""
        def _close():
            time.sleep(0.05)
            if webview.windows:
                webview.windows[0].destroy()
        threading.Thread(target=_close, daemon=True).start()


_shutdown_lock = threading.Lock()
_shutdown_done = threading.Event()


def shutdown_pipeline():
    """Stop the AI pipeline + background threads cleanly on window close.

    The uvicorn server runs in a daemon thread, so its FastAPI shutdown
    event is not guaranteed to fire when the main thread exits. We stop
    the pipeline explicitly here to release the camera and flush LINE
    alerts before the process ends.

    Idempotent + thread-safe: it is kicked onto a background thread by the
    window 'closing' handler (so the GUI thread never freezes on the several
    seconds pipeline.stop() spends joining capture threads), and is also
    called on the main thread as a safety net after the GUI loop returns. The
    lock makes the safety-net call *block until* an in-flight background
    shutdown finishes, so cleanup is guaranteed complete before the process
    exits — without ever running twice.
    """
    with _shutdown_lock:
        if _shutdown_done.is_set():
            return
        _shutdown_done.set()
        _do_shutdown()


def _do_shutdown():
    try:
        import server.api as api
        # Multi-camera: the manager owns every pipeline + its broadcaster and
        # stops them all cleanly (releases cameras, flushes threads).
        if getattr(api, "manager", None):
            api.manager.stop_all()
        try:
            from alerts.line_notify import stop_sender
            stop_sender()
        except Exception:
            pass
        print("[App] Clean shutdown complete")
    except Exception as e:
        print(f"[App] shutdown warning: {e}")


SERVER_URL = "http://127.0.0.1:7788/"


def _boot_html() -> str:
    """The small window shown while the server comes up.

    Self-contained on purpose: the logo is inlined as a data URI and there is
    no stylesheet link, because this has to render BEFORE uvicorn is listening
    — anything fetched over http would show as a broken image for the couple of
    seconds this window exists. Frameless and fixed-size, so it reads as a
    launcher rather than an application window that happens to be empty.
    """
    import base64
    logo = ""
    try:
        p = Path(__file__).parent / "ui" / "assets" / "logo-boot.png"
        logo = "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")
    except Exception:
        pass
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  html,body{{margin:0;height:100%;background:#0C0C0E;
    font-family:'Kanit','Segoe UI',sans-serif;-webkit-user-select:none;user-select:none;
    overflow:hidden;cursor:default}}
  .w{{height:100%;display:flex;flex-direction:column;align-items:center;
    justify-content:center;gap:26px;border:1px solid #232325;box-sizing:border-box}}
  img{{width:230px;height:auto;display:block}}
  /* Indeterminate: the real work is model loading, whose duration we cannot
     know, so a sweeping bar is honest where a percentage would be invented. */
  .track{{width:230px;height:2px;background:#232325;overflow:hidden;border-radius:2px}}
  .bar{{width:38%;height:100%;background:#4C8DF6;border-radius:2px;
    animation:sweep 1.15s cubic-bezier(.45,.05,.55,.95) infinite}}
  @keyframes sweep{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(363%)}}}}
  p{{margin:0;font-size:12px;color:#79797B;letter-spacing:.2px}}
  @media (prefers-reduced-motion:reduce){{.bar{{animation:none;width:100%;opacity:.5}}}}
</style></head><body><div class="w">
  {'<img src="' + logo + '" alt="ZENTRA">' if logo else '<div style="color:#EFEFF1;font-size:26px;letter-spacing:6px">ZENTRA</div>'}
  <div class="track"><div class="bar"></div></div>
  <p>กำลังเริ่มระบบ กรุณารอสักครู่...</p>
</div></body></html>"""


def _centered(w: int, h: int) -> dict:
    """x/y that put a w×h window in the middle of the primary screen.

    Uses webview.screens rather than the Win32 metrics: pywebview positions
    windows in LOGICAL units, so on a 1920x1080 display scaled to 125% the
    numbers to work from are 1536x864, not 1920x1080. Centring against the
    physical size would push the window down and to the right by a quarter of
    the screen. Returns {} when the screen cannot be read, which leaves
    pywebview's own default placement in charge.
    """
    try:
        s = webview.screens[0]
        return {"x": int(getattr(s, "x", 0) + (s.width - w) / 2),
                "y": int(getattr(s, "y", 0) + (s.height - h) / 2)}
    except Exception:
        return {}


def _wait_for_server(timeout: float = 60.0) -> bool:
    """Block until the API answers, so the main window never opens on a dead
    port. The old code slept a flat 2 s, which is under the real startup cost on
    a cold machine — the window then loaded before uvicorn was listening and
    showed a connection error instead of the app."""
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(SERVER_URL, timeout=1.5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # If the server requires a token (ZENTRA_API_TOKEN), hand it to the UI on the
    # initial URL so the same desktop window authenticates automatically. The UI
    # persists it and attaches it to every fetch/WebSocket call. No token → plain URL.
    _token = os.getenv("ZENTRA_API_TOKEN", "").strip()
    _url = SERVER_URL + (f"?token={_token}" if _token else "")

    _BOOT_W, _BOOT_H = 430, 250
    boot = webview.create_window(
        title="ZENTRA",
        html=_boot_html(),
        width=_BOOT_W, height=_BOOT_H,
        frameless=True,
        easy_drag=True,
        resizable=False,
        on_top=True,
        background_color="#0C0C0E",
        **_centered(_BOOT_W, _BOOT_H),
    )

    window = webview.create_window(
        title="ZENTRA Safety AI System",
        url=_url,
        # Open full-screen (kiosk-style) instead of a floating window. In
        # full-screen there is no native title bar / X, so the in-app quit button
        # (top-right) is the way out; F11 / toggle_fullscreen() also exits.
        # width/height are the size the window falls back to if full-screen is left.
        fullscreen=True,
        width=1280,
        height=800,
        min_size=(1024, 640),
        js_api=JsApi(),
        background_color="#0C0C0E",
        # Created hidden and revealed only once the server answers. Showing it
        # immediately would put an empty full-screen window behind the launcher
        # for the whole startup.
        hidden=True,
    )
    # Stop the pipeline as soon as the window begins closing, but do it on a
    # background thread. pipeline.stop() joins the capture/detect threads (up to
    # ~9s); running that inline on the GUI thread froze the window after the quit
    # button was pressed. Returning immediately lets the window close at once.
    def _on_closing():
        threading.Thread(target=shutdown_pipeline, daemon=True).start()

    window.events.closing += _on_closing

    def _hand_over():
        """Launcher → app. Runs on a worker thread that pywebview starts for us,
        so the GUI stays responsive while we wait on the port."""
        ok = _wait_for_server()
        if not ok:
            print("[App] ❌ server did not come up — closing launcher")
        try:
            window.show()
        except Exception as e:
            print(f"[App] show main window: {e}")
        try:
            boot.destroy()
        except Exception:
            pass

    webview.start(_hand_over, debug=False)

    # Safety net: also stop after the GUI loop returns
    shutdown_pipeline()
