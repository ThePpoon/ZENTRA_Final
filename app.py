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


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2.0)

    # If the server requires a token (ZENTRA_API_TOKEN), hand it to the UI on the
    # initial URL so the same desktop window authenticates automatically. The UI
    # persists it and attaches it to every fetch/WebSocket call. No token → plain URL.
    _token = os.getenv("ZENTRA_API_TOKEN", "").strip()
    _url = "http://127.0.0.1:7788/" + (f"?token={_token}" if _token else "")

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
        background_color="#0d1b2a",
    )
    # Stop the pipeline as soon as the window begins closing, but do it on a
    # background thread. pipeline.stop() joins the capture/detect threads (up to
    # ~9s); running that inline on the GUI thread froze the window after the quit
    # button was pressed. Returning immediately lets the window close at once.
    def _on_closing():
        threading.Thread(target=shutdown_pipeline, daemon=True).start()

    window.events.closing += _on_closing

    webview.start(debug=False)

    # Safety net: also stop after the GUI loop returns
    shutdown_pipeline()
