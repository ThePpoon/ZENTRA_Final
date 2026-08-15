/* ═══════════════════════════════════════════════
   ZENTRA — Main Application JS
   Router, WebSocket client, global state
   ═══════════════════════════════════════════════ */

/* ── Optional API auth token ──────────────────────────────────
   When the server is started with ZENTRA_API_TOKEN set, every /api and /ws call
   must carry it. The desktop app passes it once as ?token=… on the initial URL;
   we persist it (localStorage) and attach it to all fetch() + WebSocket calls.
   When the server has no token (localhost default) this whole block is a no-op. */
var ZENTRA_TOKEN = (function () {
  try {
    var t = new URL(location.href).searchParams.get('token');
    if (t) localStorage.setItem('zentra_token', t);
    return localStorage.getItem('zentra_token') || '';
  } catch (_) { return ''; }
})();
if (ZENTRA_TOKEN) {
  var _origFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    init = init || {};
    var headers = new Headers(
      init.headers || (input && typeof input !== 'string' && input.headers) || {}
    );
    if (!headers.has('Authorization')) headers.set('Authorization', 'Bearer ' + ZENTRA_TOKEN);
    init.headers = headers;
    return _origFetch(input, init);
  };
}

const ZENTRA = {
  /* ── State ──────────────────────────────────── */
  state: {
    pipeline: { running: false, source: null },
    // Start on 'standby' (neutral grey), NOT 'ok' (green). Green before the pipeline
    // has reported anything falsely reads as "all modules running / everyone safe";
    // the real status arrives over the WS on connect and corrects this.
    modules:  { ppe: 'standby', zone: 'standby', fall: 'standby' },
    alerts:   { total: 0, warning: 0, alert: 0, emergency: 0 },
    uptime:   0,
    last_emergency: null,
    camera_label: 'กล้อง #1',
    camera:   'disconnected',   // connected | reconnecting | disconnected
    recentAlarms: [],           // [{level,message,time,camera}] newest first
  },

  ws: null,
  _wsRetryTimer: null,
  _statusTimer:  null,
  _currentScreen: null,

  /* ── Router ─────────────────────────────────── */
  async navigate(screenId, params = {}) {
    try {
      const res       = await fetch(`/ui/screens/${screenId}.html`);
      const html      = await res.text();
      const container = document.getElementById('app');

      // Dismiss any modal still standing. Dialogs live in #zt-ui-root on the
      // BODY, not inside #app, so replacing the screen below leaves an orphaned
      // .zt-overlay behind — a full-screen backdrop at z-index 4000 that
      // silently swallows every click and keystroke on whatever screen loads
      // next. The screen looks rendered and completely dead.
      document.querySelectorAll('.zt-overlay').forEach(o => o.remove());

      container.innerHTML = html;
      // Subtle fade-in on every navigation so screens ease in instead of
      // hard-cutting. Opacity only (no transform) so it never becomes a
      // containing block for a screen's fixed modals/toasts. Re-trigger the
      // animation by removing the class + forcing a reflow.
      container.classList.remove('zt-screen-enter');
      void container.offsetWidth;
      container.classList.add('zt-screen-enter');
      ZENTRA._currentScreen = screenId;

      // Left sidebar for main screens; full-screen for splash/source
      const SHELL = ['dashboard', 'zone_editor', 'history', 'settings', 'cameras'];
      if (SHELL.includes(screenId)) ZENTRA.mountSidebar(screenId);
      else ZENTRA.hideSidebar();

      // innerHTML does NOT auto-execute <script> tags — re-create them.
      // Re-creating a <script> element makes the browser execute it in
      // GLOBAL scope (eval() would only define functions locally, so
      // window['init_<screen>'] would never be found).
      const scripts = container.querySelectorAll('script');
      const externalLoads = [];

      for (const oldScript of scripts) {
        if (oldScript.src) {
          // External CDN script — load once, append to head, await onload
          if (!document.querySelector(`script[data-cdn="${oldScript.src}"]`)) {
            externalLoads.push(new Promise(resolve => {
              const el = document.createElement('script');
              el.src   = oldScript.src;
              el.setAttribute('data-cdn', oldScript.src);
              el.onload  = resolve;
              el.onerror = resolve;
              document.head.appendChild(el);
            }));
          }
        } else if (oldScript.textContent.trim()) {
          // Inline script — re-create so it runs in global scope
          const el = document.createElement('script');
          el.textContent = oldScript.textContent;
          document.body.appendChild(el);
          document.body.removeChild(el);
        }
      }

      // Wait for external scripts (e.g. Chart.js) before calling init
      if (externalLoads.length) await Promise.all(externalLoads);

      // Call the screen's init function (now globally defined)
      const fn = window[`init_${screenId}`];
      if (typeof fn === 'function') fn(params);

      // Keep the header clock/status pill running on screens that have a navbar
      ZENTRA.startHeaderClock();
    } catch (e) {
      console.error('[ZENTRA] navigate error:', screenId, e);
    }
  },

  /* ── WebSocket ──────────────────────────────── */
  connectWS() {
    if (ZENTRA.ws && ZENTRA.ws.readyState < 2) return;

    // Derive from the page origin so a remote/LAN browser (and HTTPS) works —
    // a hardcoded 127.0.0.1 only ever connected on the host running the server.
    const _wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // The WS handshake can't send an Authorization header, so pass the token as a
    // query param (server gates /ws/stream on it). Empty when auth is disabled.
    const _wsQuery = ZENTRA_TOKEN ? ('?token=' + encodeURIComponent(ZENTRA_TOKEN)) : '';
    const ws = new WebSocket(_wsProto + '//' + location.host + '/ws/stream' + _wsQuery);
    ZENTRA.ws = ws;

    ws.onopen  = () => { clearTimeout(ZENTRA._wsRetryTimer); };
    ws.onclose = () => { ZENTRA._wsRetryTimer = setTimeout(() => ZENTRA.connectWS(), 2000); };
    ws.onerror = () => { ws.close(); };

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        ZENTRA._handleWsMsg(msg);
      } catch (_) {}
    };
  },

  _handleWsMsg(msg) {
    if (msg.type === 'frame') {
      // Multi-camera: every frame is tagged with its camera_id so N cameras can
      // share this ONE WebSocket. We keep a per-camera map — the Cameras grid and
      // the Dashboard live panel both read it by id — plus the legacy
      // single-stream fields for any screen still bound to one #video-feed.
      const cid = msg.camera_id || 'cam0';
      const dataUrl = 'data:image/jpeg;base64,' + msg.data;
      if (!ZENTRA._frames)  ZENTRA._frames  = {};
      if (!ZENTRA._frameAt) ZENTRA._frameAt = {};
      ZENTRA._frames[cid]  = dataUrl;
      ZENTRA._frameAt[cid] = Date.now();
      if (!ZENTRA._primaryCam) ZENTRA._primaryCam = cid;
      if (cid === ZENTRA._primaryCam) {
        ZENTRA._lastFrame = dataUrl;
        ZENTRA._lastFrameAt = Date.now();
        const el = document.getElementById('video-feed');
        if (el) el.src = dataUrl;
      }
      // Fast path: if a grid tile for this camera exists, update it directly.
      const tile = document.getElementById('cam-tile-' + cid);
      if (tile) {
        tile.src = dataUrl;
        // First frame for this camera → swap the "ไม่มีสัญญาณ" placeholder for
        // the picture right now. Waiting for the 2.5s status poll to notice left
        // a live camera looking dead for up to two and a half seconds.
        if (tile.style.display === 'none') {
          tile.style.display = 'block';
          const off = document.getElementById('cam-off-' + cid);
          if (off) off.style.display = 'none';
        }
      }
      // Per-screen frame hook (CCTV grid / fullscreen use it).
      if (typeof ZENTRA._frameHook === 'function') ZENTRA._frameHook(cid, dataUrl);
    }
    if (msg.type === 'event') {
      if (msg.event === 'status' || msg.modules) {
        if (msg.modules)  ZENTRA.state.modules  = msg.modules;
        if (msg.alerts)   ZENTRA.state.alerts   = msg.alerts;
        if (msg.camera)   ZENTRA.state.camera   = msg.camera;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
        ZENTRA._updateCameraState();
        ZENTRA._reportEngineError(msg.engine_error);
      }
      if (msg.event === 'alert') {
        const lvl = msg.level || 'warning';
        // Prefer authoritative counts from server; fall back to local increment
        if (msg.alerts) {
          ZENTRA.state.alerts = msg.alerts;
        } else {
          ZENTRA.state.alerts.total++;
          if (lvl === 'warning' || lvl === 'alert' || lvl === 'emergency') {
            ZENTRA.state.alerts[lvl] = (ZENTRA.state.alerts[lvl] || 0) + 1;
          }
        }
        // Push to the recent-alarms list (newest first, cap 30)
        ZENTRA.state.recentAlarms.unshift({
          level: lvl, message: msg.message || '', time: msg.timestamp || '', camera: msg.camera || '',
        });
        ZENTRA.state.recentAlarms = ZENTRA.state.recentAlarms.slice(0, 30);
        if (lvl === 'emergency') {
          ZENTRA.state.last_emergency = msg;
        }
        ZENTRA._updateAlertCounters();
        ZENTRA._updateKPIs();
        ZENTRA._renderAlarms();
        ZENTRA._updateModuleStatus();
        // Dashboard toast hook (only fires when dashboard is loaded)
        if (typeof ZENTRA._toastHook === 'function') ZENTRA._toastHook(msg);
        // CCTV grid hook: flash the tile of the camera that raised the alert.
        if (typeof ZENTRA._gridAlertHook === 'function') ZENTRA._gridAlertHook(msg);
      }
    }
  },

  /* ── UI Update Helpers ──────────────────────── */
  _updateModuleStatus() {
    const map = { ppe: 'PPE Module', zone: 'Zone Module', fall: 'Fall Module' };
    const m   = ZENTRA.state.modules;
    // ok = green, off/standby = neutral grey (not a red error), anything else = err
    const cssFor = (st) => st === 'ok' ? 'ok' : (st === 'off' || st === 'standby') ? 'off' : 'err';
    const thFor  = (st) => st === 'ok' ? 'ปกติ' : st === 'off' ? 'ปิดอยู่'
                         : st === 'standby' ? 'สแตนด์บาย' : 'ไม่ปกติ';
    for (const [key, _label] of Object.entries(map)) {
      const dotEl   = document.getElementById(`dot-${key}`);
      const labelEl = document.getElementById(`lbl-${key}`);
      if (!dotEl) continue;
      const st  = m[key];
      const cls = cssFor(st);
      dotEl.className = 'status-dot ' + cls;
      if (labelEl) {
        labelEl.textContent = thFor(st);
        labelEl.className   = 'module-label ' + cls;
      }
    }
  },

  // The "warning" tile means "everything that is not an emergency" — PPE
  // (warning) plus zone intrusions (alert). The emergency tile counts ONLY
  // emergencies, so it stays at 0 until somebody actually falls.
  _nonEmergency(a) { return (a.warning || 0) + (a.alert || 0); },

  /* Is a live video view on screen right now? Any of the three containers that
     render frames counts. Used by the frame watchdog, which must not be tied to
     one screen's markup. */
  _hasLiveView() {
    return !!(document.getElementById('dash-grid')
           || document.getElementById('cam-grid')
           || document.getElementById('video-feed'));
  },

  /* The AI engine failed to load → the video is a plain passthrough and NOTHING is
     being detected. Silence here would look identical to "everyone is compliant",
     so say it out loud, once per distinct reason. */
  _reportEngineError(err) {
    var existing = document.getElementById('zt-engine-error');
    // Engine recovered (or never failed) → remove the banner and reset both flags
    // (a future error, even the same text, should show again).
    if (!err) {
      if (existing) existing.remove();
      ZENTRA._engineErrShown = null;
      ZENTRA._engineErrDismissed = null;
      return;
    }
    // User closed the banner for THIS reason → keep it hidden until the reason
    // changes or clears. A different error text still surfaces.
    if (ZENTRA._engineErrDismissed === err) return;
    if (ZENTRA._engineErrShown === err && existing) return;
    ZENTRA._engineErrShown = err;
    console.error('[ZENTRA] engine not loaded:', err);
    // A PERSISTENT bar, not a 15s toast: while the AI engine is down the video is a
    // plain passthrough and NOTHING is detected — clean video with no boxes looks
    // exactly like "everyone is compliant", so the warning must stay on screen until
    // it is actually fixed (or the user explicitly dismisses it).
    var bar = existing || document.createElement('div');
    bar.id = 'zt-engine-error';
    bar.setAttribute('role', 'alert');
    // Bottom (not top) so it doesn't cover the topbar; leaves room for the close ✕.
    bar.style.cssText = 'position:fixed;bottom:0;left:0;right:0;z-index:9999;'
      + 'background:var(--red,#ef4444);color:#fff;padding:9px 44px 9px 16px;font-size:13px;'
      + 'font-weight:600;text-align:center;box-shadow:0 -2px 10px rgba(0,0,0,.35);'
      + 'font-family:var(--font)';
    // err is a backend exception string — treat as text, not markup.
    bar.innerHTML = '<span></span>'
      + '<button type="button" title="ปิด" onclick="ZENTRA._dismissEngineError()" '
      + 'style="position:absolute;top:50%;right:12px;transform:translateY(-50%);'
      + 'width:24px;height:24px;border:none;background:rgba(255,255,255,.18);color:#fff;'
      + 'border-radius:6px;cursor:pointer;font-size:15px;line-height:1;display:flex;'
      + 'align-items:center;justify-content:center">×</button>';
    bar.firstChild.textContent = '⚠ ระบบ AI ไม่ทำงาน — ไม่มีการตรวจจับใด ๆ: ' + err;
    if (!existing) document.body.appendChild(bar);
  },

  // Close the engine-error banner and remember the dismissed reason so the next
  // status poll doesn't immediately re-open it.
  _dismissEngineError() {
    var bar = document.getElementById('zt-engine-error');
    if (bar) bar.remove();
    ZENTRA._engineErrDismissed = ZENTRA._engineErrShown;
    ZENTRA._engineErrShown = null;
  },

  _updateAlertCounters() {
    var a = ZENTRA.state.alerts;
    var el = function(id) { return document.getElementById(id); };
    if (el('cnt-total'))     el('cnt-total').textContent     = a.total;
    if (el('cnt-warning'))   el('cnt-warning').textContent   = ZENTRA._nonEmergency(a);
    if (el('cnt-emergency')) el('cnt-emergency').textContent = a.emergency || 0;
    // Topbar notification badge
    var badge = el('topbar-badge');
    if (badge) {
      var total = a.total || 0;
      badge.textContent = total > 99 ? '99+' : total;
      if (total > 0) badge.classList.add('visible');
      else badge.classList.remove('visible');
    }
  },

  _fmtUptime(secs) {
    secs = secs || 0;
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    return h > 0 ? `${h}:${String(m).padStart(2,'0')}` : `${m} น.`;
  },

  _updateKPIs() {
    const a = ZENTRA.state.alerts || {};
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('kpi-total',     a.total     || 0);
    // One counter PER LEVEL. warning = PPE, alert = zone intrusion,
    // emergency = fall. These are distinct incident kinds, so summing warning
    // and alert into one number (what this used to do) made every zone
    // intrusion invisible on the dashboard.
    set('kpi-warning',   a.warning   || 0);
    set('kpi-alert',     a.alert     || 0);
    set('kpi-emergency', a.emergency || 0);
    set('kpi-uptime',    ZENTRA._fmtUptime(ZENTRA.state.uptime));
    const m = ZENTRA.state.modules || {};
    const ok = ['ppe','zone','fall'].filter(k => m[k] === 'ok').length;
    set('kpi-modules', `${ok}/3`);
    // AI-modules subtitle + dot (honest: reflects real active/standby state)
    const camConnected = (ZENTRA.state.camera === 'connected');
    const subEl = document.getElementById('kpi-modules-sub');
    const mdot  = document.getElementById('kpi-modules-dot');
    if (subEl) subEl.textContent = ok > 0 ? (ok + ' โมดูลทำงาน') : (camConnected ? 'สแตนด์บาย' : 'ยังไม่เริ่ม');
    if (mdot)  mdot.style.background = ok > 0 ? 'var(--green)' : 'var(--text-muted)';
    // Camera-online count: RUNNING cameras out of CONFIGURED cameras. This was
    // `camConnected ? 1 : 0` over a hardcoded "/1", so a site running three
    // cameras read "1/1" — it reported whether the primary camera was up, not
    // how many cameras were.
    const camsOn    = ZENTRA.state.camerasActive;
    const camsTotal = ZENTRA.state.camerasTotal;
    set('kpi-cam-online', camsOn != null ? camsOn : (camConnected ? 1 : 0));
    set('kpi-cam-total',  camsTotal != null ? camsTotal : 1);
    // Color tiles only when there is something to show (control-room style)
    const wt = document.getElementById('kpi-tile-warning');
    const et = document.getElementById('kpi-tile-emergency');
    if (wt) wt.classList.toggle('warn',  (a.warning   || 0) > 0);
    if (et) et.classList.toggle('alarm', (a.emergency || 0) > 0);
  },

  _renderAlarms() {
    const list = document.getElementById('alarm-list');
    if (!list) return;
    const items = ZENTRA.state.recentAlarms || [];
    if (!items.length) {
      list.innerHTML = '<div style="text-align:center;padding:24px 0;color:var(--text-muted);font-size:13px">ยังไม่มีการแจ้งเตือน</div>';
      return;
    }
    const colorMap = { emergency: 'var(--red)', alert: 'var(--orange)', warning: 'var(--yellow)', info: 'var(--accent)' };
    const bgMap    = { emergency: 'var(--red-dim)', alert: 'rgba(255,122,69,.12)', warning: 'rgba(245,158,11,.12)', info: 'var(--accent-dim)' };
    list.innerHTML = items.slice(0, 8).map(function(it) {
      var lvl = it.level || 'alert';
      var col = colorMap[lvl] || 'var(--accent)';
      var bg  = bgMap[lvl]    || 'var(--accent-dim)';
      return '<div style="display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid var(--border)">'
        + '<span style="width:8px;height:8px;border-radius:50%;background:' + col + ';flex-shrink:0;margin-top:5px"></span>'
        + '<div style="flex:1;min-width:0">'
        + '<div style="font-size:13px;font-weight:600;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + ZENTRA.esc(it.message) + '</div>'
        + '<div style="font-size:11.5px;color:var(--text-muted);margin-top:1px">' + ZENTRA.esc(it.time) + (it.camera ? ' · ' + ZENTRA.esc(it.camera) : '') + '</div>'
        + '</div>'
        + '<span style="padding:2px 8px;border-radius:99px;background:' + bg + ';color:' + col + ';font-size:10.5px;font-weight:700;white-space:nowrap;flex-shrink:0">' + lvl.toUpperCase() + '</span>'
        + '</div>';
    }).join('');
  },

  _updateCameraState() {
    // Show a connecting/reconnecting overlay over the video feed.
    const overlay = document.getElementById('video-overlay');
    if (overlay) {
      const state = ZENTRA.state.camera;
      if (state === 'connected') {
        overlay.classList.add('hidden');
      } else {
        overlay.classList.remove('hidden');
        const txt = overlay.querySelector('.video-overlay-text');
        if (txt) {
          txt.textContent = (state === 'reconnecting')
            ? 'สัญญาณกล้องหลุด — กำลังเชื่อมต่อใหม่...'
            : 'กำลังเชื่อมต่อกล้อง...';
        }
      }
    }
    // Camera status dot hook (cameras.html updates its own dot)
    if (typeof ZENTRA._camDotHook === 'function') ZENTRA._camDotHook();
  },

  /* ── Status Poll ─────────────────────────────── */
  startStatusPoll() {
    // Clear any existing poll first so repeated dashboard visits don't
    // stack multiple intervals (would multiply /api/status traffic).
    if (ZENTRA._statusTimer) clearInterval(ZENTRA._statusTimer);
    ZENTRA._statusTimer = setInterval(async () => {
      try {
        const res  = await fetch('/api/status');
        const data = await res.json();
        ZENTRA.state.modules = data.modules ?? ZENTRA.state.modules;
        ZENTRA.state.alerts  = data.alerts  ?? ZENTRA.state.alerts;
        ZENTRA.state.uptime  = data.uptime  ?? 0;
        ZENTRA.state.camerasActive = data.cameras_active ?? null;
        ZENTRA.state.camerasTotal  = data.cameras_total  ?? null;
        if (data.camera) ZENTRA.state.camera = data.camera;
        ZENTRA._updateModuleStatus();
        ZENTRA._updateAlertCounters();
        ZENTRA._updateKPIs();
        ZENTRA._updateCameraState();
        if (data.running) ZENTRA._reportEngineError(data.engine_error);

        // Watchdog: pipeline running but no video frame for >6s → the WS
        // stalled; force a reconnect so the live view self-heals. Checks any
        // live-view container, not just the dashboard's old single #video-feed
        // <img> — that element no longer exists (the dashboard shows a grid
        // now), which would have silently disabled the watchdog everywhere.
        if (data.running && ZENTRA._hasLiveView()) {
          const since = Date.now() - (ZENTRA._lastFrameAt || 0);
          if (since > 6000) {
            try { if (ZENTRA.ws) ZENTRA.ws.close(); } catch (_) {}
            ZENTRA.ws = null;
            ZENTRA.connectWS();
            ZENTRA._lastFrameAt = Date.now();   // grace period before next retry
          }
        }

        // Nothing running → drop the frozen last frames so the panel goes empty
        // rather than showing a still image of a camera that is now offline,
        // which reads as a working feed that simply has nobody in view.
        if (!data.running) {
          const vf = document.getElementById('video-feed');
          if (vf && vf.getAttribute('src')) vf.src = '';
          ZENTRA._lastFrame = null;
          ZENTRA._frames = {};
          document.querySelectorAll('.dash-tile img, .cam-tile-img').forEach(img => {
            img.removeAttribute('src');
            img.style.display = 'none';
            const off = document.getElementById(
              img.id.replace(/^dash-tile-/, 'dash-off-').replace(/^cam-tile-/, 'cam-off-'));
            if (off) off.style.display = 'flex';
          });
        }
      } catch (_) {}
    }, 2000);
  },

  stopStatusPoll() { clearInterval(ZENTRA._statusTimer); },

  /* ── Navigation helpers (called from HTML) ───── */
  goTo(screen) { ZENTRA.navigate(screen); },

  /* ── Bootstrap ──────────────────────────────── */
  async init() {
    ZENTRA.navigate('splash');
  },
};

/* ─── ZENTRA dialogs / toasts / date picker (theme-native — replaces browser
   alert()/confirm()/prompt() and the native calendar) ─────────────────────── */
ZENTRA._uiRoot = function () {
  var r = document.getElementById('zt-ui-root');
  if (!r) { r = document.createElement('div'); r.id = 'zt-ui-root'; document.body.appendChild(r); }
  return r;
};

/* Escape untrusted text before it goes anywhere near innerHTML. Alert messages
   carry operator-supplied zone names, so they are data, not markup. Escaping only
   "<" leaves attribute and quote contexts open. */
ZENTRA.esc = function (s) {
  return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
    return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
  });
};

/* Set an <img>.src for a URL that may require the API token. A plain <img src>
   can't send an Authorization header, so when a token is set and the URL is an
   /api/* resource, fetch the bytes (the fetch override adds the token) and use a
   blob URL. No token → set src directly, unchanged. */
ZENTRA.setImgSrc = function (img, url) {
  if (!img) return;
  var revoke = function () { if (img._zurl) { try { URL.revokeObjectURL(img._zurl); } catch (_) {} img._zurl = null; } };
  if (!ZENTRA_TOKEN || !/^\/api\//.test(url)) { revoke(); img.src = url; return; }
  fetch(url).then(function (r) { return r.ok ? r.blob() : Promise.reject(r.status); })
    .then(function (b) { revoke(); img._zurl = URL.createObjectURL(b); img.src = img._zurl; })
    .catch(function () { revoke(); img.src = ''; });
};

/* Download an /api resource through the native save bridge instead of
   `window.location = url`. A raw navigation points the whole WebView at a binary
   endpoint — WKWebView/WebView2 can't render a PDF/CSV inline and has no download
   handler, so the SPA is replaced by a blank white page (and, when a token is
   set, the navigation carries no Authorization header → 401 → blank too).
   Here we fetch the bytes (the fetch override attaches the token), then hand them
   to pywebview's save_binary, which pops a real native "Save as" dialog. In a
   plain browser (no pywebview) fall back to a normal anchor download. */
ZENTRA.saveResource = async function (url, filename) {
  var hasBridge = window.pywebview && window.pywebview.api && window.pywebview.api.save_binary;
  if (!hasBridge) {
    var a = document.createElement('a');
    a.href = url; a.download = filename || '';
    document.body.appendChild(a); a.click(); a.remove();
    return;
  }
  try {
    var r = await fetch(url);
    if (!r.ok) throw r.status;
    var bytes = new Uint8Array(await r.arrayBuffer());
    // arrayBuffer → base64 in chunks (String.fromCharCode.apply blows the call
    // stack on large reports if handed the whole array at once).
    var bin = '', CHUNK = 0x8000;
    for (var i = 0; i < bytes.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    var saved = await window.pywebview.api.save_binary(btoa(bin), filename);
    if (saved) ZENTRA.toast('บันทึกไฟล์แล้ว', 'success');
  } catch (_) {
    ZENTRA.toast('บันทึกไฟล์ไม่สำเร็จ', 'error');
  }
};

ZENTRA.toast = function (message, type, ms) {
  type = type || 'info'; ms = ms || 3200;
  var root = ZENTRA._uiRoot();
  var wrap = document.getElementById('zt-toasts');
  if (!wrap) { wrap = document.createElement('div'); wrap.id = 'zt-toasts'; wrap.className = 'zt-toasts'; root.appendChild(wrap); }
  var el = document.createElement('div');
  el.className = 'zt-toast zt-' + type;
  el.innerHTML = '<span class="zt-toast-dot"></span><span>' + ZENTRA.esc(message) + '</span>';
  wrap.appendChild(el);
  requestAnimationFrame(function () { el.classList.add('in'); });
  setTimeout(function () { el.classList.remove('in'); setTimeout(function () { el.remove(); }, 220); }, ms);
};

ZENTRA.dialog = function (opts) {
  opts = opts || {};
  return new Promise(function (resolve) {
    var root = ZENTRA._uiRoot();
    var ov = document.createElement('div');
    ov.className = 'zt-overlay';
    var hasInput = !!opts.input;
    var hasCancel = opts.cancel !== false;
    ov.innerHTML =
      '<div class="zt-modal" role="dialog" aria-modal="true">'
      + (opts.title ? '<div class="zt-modal-title">' + opts.title + '</div>' : '')
      + (opts.message ? '<div class="zt-modal-msg">' + String(opts.message).replace(/</g, '&lt;').replace(/\n/g, '<br>') + '</div>' : '')
      + (hasInput ? '<input class="zt-modal-input" type="text">' : '')
      + '<div class="zt-modal-actions">'
      + (hasCancel ? '<button class="zt-btn zt-btn-ghost" data-act="cancel">' + (opts.cancelText || 'ยกเลิก') + '</button>' : '')
      + '<button class="zt-btn ' + (opts.danger ? 'zt-btn-danger' : 'zt-btn-primary') + '" data-act="ok">' + (opts.okText || 'ตกลง') + '</button>'
      + '</div></div>';
    root.appendChild(ov);
    var input = ov.querySelector('.zt-modal-input');
    if (input) { input.value = opts.defaultValue != null ? opts.defaultValue : ''; if (opts.placeholder) input.placeholder = opts.placeholder; }
    requestAnimationFrame(function () { ov.classList.add('in'); if (input) { input.focus(); input.select(); } });
    function close(val) {
      ov.classList.remove('in');
      document.removeEventListener('keydown', onKey);
      setTimeout(function () { ov.remove(); }, 170);
      resolve(val);
    }
    function onKey(e) {
      if (e.key === 'Escape' && hasCancel) close(hasInput ? null : false);
      else if (e.key === 'Enter') { e.preventDefault(); close(hasInput ? (input ? input.value : true) : true); }
    }
    document.addEventListener('keydown', onKey);
    ov.addEventListener('click', function (e) {
      var act = e.target.getAttribute && e.target.getAttribute('data-act');
      if (e.target === ov && hasCancel) return close(hasInput ? null : false);
      if (act === 'ok') close(hasInput ? (input ? input.value : true) : true);
      else if (act === 'cancel') close(hasInput ? null : false);
    });
  });
};

ZENTRA.alert   = function (message, title) {
  return ZENTRA.dialog({ title: title || 'แจ้งเตือน', message: message, cancel: false, okText: 'ตกลง' });
};
ZENTRA.confirm = function (message, opts) {
  opts = opts || {};
  return ZENTRA.dialog({ title: opts.title || 'ยืนยัน', message: message,
    okText: opts.okText || 'ยืนยัน', cancelText: opts.cancelText || 'ยกเลิก', danger: opts.danger })
    .then(function (v) { return v === true; });
};
ZENTRA.prompt  = function (message, defaultValue, opts) {
  opts = opts || {};
  return ZENTRA.dialog({ title: opts.title || message, input: true, defaultValue: defaultValue,
    placeholder: opts.placeholder, okText: opts.okText || 'ตกลง' });
};

// Themed calendar popup. current = 'YYYY-MM-DD' | ''; onPick(v): v is
// 'YYYY-MM-DD' | '' (cleared) | undefined (dismissed, no change).
ZENTRA.datePicker = function (anchorEl, current, onPick) {
  var ex = document.getElementById('zt-datepop'); if (ex) ex.remove();
  var root = ZENTRA._uiRoot();
  var pop = document.createElement('div'); pop.id = 'zt-datepop'; pop.className = 'zt-datepop';
  function ymd(x) { return x.getFullYear() + '-' + String(x.getMonth() + 1).padStart(2, '0') + '-' + String(x.getDate()).padStart(2, '0'); }
  var base = current ? new Date(current + 'T00:00:00') : new Date();
  if (isNaN(base)) base = new Date();
  var view = new Date(base.getFullYear(), base.getMonth(), 1);
  var monthsTh = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน', 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม'];
  function render() {
    var y = view.getFullYear(), m = view.getMonth();
    var first = new Date(y, m, 1).getDay(), days = new Date(y, m + 1, 0).getDate(), todayS = ymd(new Date());
    var html = '<div class="zt-dp-head"><button data-nav="-1" class="zt-dp-nav">‹</button>'
      + '<span class="zt-dp-title">' + monthsTh[m] + ' ' + (y + 543) + '</span>'
      + '<button data-nav="1" class="zt-dp-nav">›</button></div><div class="zt-dp-grid">';
    ['อา', 'จ', 'อ', 'พ', 'พฤ', 'ศ', 'ส'].forEach(function (w) { html += '<span class="zt-dp-dow">' + w + '</span>'; });
    for (var i = 0; i < first; i++) html += '<span></span>';
    for (var day = 1; day <= days; day++) {
      var ds = y + '-' + String(m + 1).padStart(2, '0') + '-' + String(day).padStart(2, '0');
      html += '<button class="zt-dp-day' + (ds === current ? ' sel' : '') + (ds === todayS ? ' today' : '') + '" data-date="' + ds + '">' + day + '</button>';
    }
    html += '</div><div class="zt-dp-foot"><button data-clear="1" class="zt-dp-mini">ล้าง</button><button data-today="1" class="zt-dp-mini">วันนี้</button></div>';
    pop.innerHTML = html;
  }
  render();
  root.appendChild(pop);
  var r = anchorEl.getBoundingClientRect();
  pop.style.left = Math.max(8, Math.min(r.left, window.innerWidth - 268)) + 'px';
  pop.style.top  = (r.bottom + 6) + 'px';
  requestAnimationFrame(function () { pop.classList.add('in'); });
  function done(val) { pop.remove(); document.removeEventListener('mousedown', outside, true); if (onPick) onPick(val); }
  function outside(e) { if (!pop.contains(e.target) && e.target !== anchorEl) done(undefined); }
  setTimeout(function () { document.addEventListener('mousedown', outside, true); }, 0);
  pop.addEventListener('click', function (e) {
    var t = e.target, nav = t.getAttribute('data-nav');
    if (nav) { view.setMonth(view.getMonth() + parseInt(nav, 10)); render(); return; }
    var ds = t.getAttribute('data-date'); if (ds) return done(ds);
    if (t.getAttribute('data-clear')) return done('');
    if (t.getAttribute('data-today')) return done(ymd(new Date()));
  });
};

/* ─── Left Sidebar (persistent, lives in <body> so it survives #app swaps) ─── */
// Clean inline SVG line icons (Lucide-style) — sharp + theme-aware (inherit
// currentColor), far more professional than emoji which render inconsistently.
ZENTRA.icons = {
  dashboard: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
  zone:      '<path d="M14.106 5.553a2 2 0 0 0 1.788 0l3.659-1.83A1 1 0 0 1 21 4.619v12.764a1 1 0 0 1-.553.894l-4.553 2.277a2 2 0 0 1-1.788 0l-4.212-2.106a2 2 0 0 0-1.788 0l-3.659 1.83A1 1 0 0 1 3 19.381V6.618a1 1 0 0 1 .553-.894l4.553-2.277a2 2 0 0 1 1.788 0z"/><path d="M15 5.764v15"/><path d="M9 3.236v15"/>',
  history:   '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>',
  settings:  '<path d="M20 7h-9"/><path d="M14 17H5"/><circle cx="17" cy="17" r="3"/><circle cx="7" cy="7" r="3"/>',
  shield:    '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
  cameras:   '<path d="m22 8-6 4 6 4V8Z"/><rect width="14" height="12" x="2" y="6" rx="2"/>',
};
ZENTRA.icon = function (name) {
  return '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
       + 'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
       + (ZENTRA.icons[name] || '') + '</svg>';
};

function renderSidebar(active) {
  var items = [
    { id: 'dashboard',   ico: 'dashboard', label: 'แดชบอร์ด'    },
    { id: 'cameras',     ico: 'cameras',   label: 'กล้อง'       },
    { id: 'zone_editor', ico: 'zone',      label: 'แก้ไขพื้นที่' },
    { id: 'history',     ico: 'history',   label: 'ประวัติ'     },
    { id: 'settings',    ico: 'settings',  label: 'ตั้งค่า'      },
  ];
  var nav = items.map(function (it) {
    return '<button class="sb-item' + (it.id === active ? ' active' : '') + '"'
      + ' onclick="ZENTRA.navigate(\'' + it.id + '\')"'
      + ' aria-label="' + it.label + '">'
      + '<span class="sb-ico">' + ZENTRA.icon(it.ico) + '</span>'
      + '<span class="sb-tooltip">' + it.label + '</span>'
      + '</button>';
  }).join('');

  return '<div class="sb-brand">' + ZENTRA.icon('shield') + '</div>'
    + '<nav class="sb-nav">' + nav + '</nav>';
}

function renderTopbar() {
  var isDark = (document.body.getAttribute('data-theme') || 'dark') === 'dark';
  var themeIco = isDark
    ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>'
    : '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>';
  return '<div id="app-topbar">'
    + '<div class="topbar-left">'
    + '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    + '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/>'
    + '<circle cx="12" cy="10" r="3"/></svg>'
    + '<span class="topbar-site">ZENTRA &middot; โรงงาน</span>'
    + '</div>'
    + '<div class="topbar-right">'
    // Clock + system state used to sit at the foot of the icon rail, where
    // 68px forced them to 10px type. In the header they are readable and sit
    // where a VMS operator expects the machine's state to be.
    + '<span class="sys-pill ok" id="sys-pill"><span class="sys-dot"></span>'
    +   '<span id="sys-pill-text">ปกติ</span></span>'
    + '<span class="nav-clock" id="nav-clock">--:--:--</span>'
    + '<span class="topbar-sep"></span>'
    // Lock badge: shown only when the server requires a token (ZENTRA_API_TOKEN),
    // so the operator can see at a glance that the API is protected.
    + (ZENTRA_TOKEN
        ? '<span title="API ป้องกันด้วยโทเคน (ZENTRA_API_TOKEN)" style="display:inline-flex;align-items:center;gap:5px;height:34px;padding:0 10px;border-radius:8px;border:1px solid var(--border);background:var(--green-dim);color:var(--green);font-size:11.5px;font-weight:700">'
          + '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>ป้องกัน</span>'
        : '')
    + '<button class="topbar-btn" id="topbar-theme-btn" onclick="ZENTRA.toggleTheme()" title="สลับธีม">'
    + '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + themeIco + '</svg>'
    + '</button>'
    + '<button class="topbar-btn topbar-bell" onclick="ZENTRA.navigate(\'history\')" title="การแจ้งเตือน · เปิดหน้าประวัติ">'
    + '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    + '<path d="M10.268 21a2 2 0 0 0 3.464 0"/>'
    + '<path d="M3.262 15.326A1 1 0 0 0 4 17h16a1 1 0 0 0 .74-1.673C19.41 13.956 18 12.499 18 8A6 6 0 0 0 6 8c0 4.499-1.411 5.956-2.738 7.326"/></svg>'
    + '<span class="topbar-badge" id="topbar-badge"></span>'
    + '</button>'
    // Divider + quit: closes the app cleanly (with confirm). Redundant with the
    // native window X in windowed mode, but the reachable exit in fullscreen/kiosk.
    + '<span style="width:1px;height:20px;background:var(--border);margin:0 2px"></span>'
    + '<button class="topbar-btn" id="topbar-quit-btn" onclick="ZENTRA.quitApp()" title="ปิดโปรแกรม">'
    + '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    + '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/></svg>'
    + '</button>'
    + '</div>'
    + '</div>';
}

// Quit the desktop app (with confirm). The pywebview JsApi releases the camera
// + pipeline cleanly on destroy; in a plain browser (no pywebview) there's
// nothing to close, so tell the user to use the window controls instead.
ZENTRA.quitApp = async function () {
  var ok = await ZENTRA.confirm('ปิดโปรแกรม ZENTRA', {
    danger: true, okText: 'ปิดโปรแกรม',
  });
  if (!ok) return;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.close_app) {
      window.pywebview.api.close_app();
    } else {
      ZENTRA.toast('โปรดปิดจากปุ่มหน้าต่างของระบบ', 'error');
    }
  } catch (_) {
    ZENTRA.toast('ปิดโปรแกรมไม่สำเร็จ', 'error');
  }
};

ZENTRA.mountSidebar = function (active) {
  var sb = document.getElementById('app-sidebar');
  if (!sb) {
    sb = document.createElement('aside');
    sb.id = 'app-sidebar';
    document.body.appendChild(sb);
  }
  sb.innerHTML = renderSidebar(active);
  sb.style.display = 'flex';
  document.body.classList.add('with-sidebar');

  // Mount topbar once; update theme btn on re-navigation
  if (!document.getElementById('app-topbar')) {
    var div = document.createElement('div');
    div.innerHTML = renderTopbar();
    document.body.appendChild(div.firstElementChild);
  } else {
    ZENTRA._updateThemeBtn();
  }
};

ZENTRA.hideSidebar = function () {
  var sb = document.getElementById('app-sidebar');
  if (sb) sb.style.display = 'none';
  document.body.classList.remove('with-sidebar');
  var tb = document.getElementById('app-topbar');
  if (tb) tb.remove();
};

ZENTRA.toggleTheme = function () {
  var cur = document.body.getAttribute('data-theme') || 'dark';
  document.body.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
  ZENTRA._updateThemeBtn();
};

ZENTRA._updateThemeBtn = function () {
  var btn = document.getElementById('topbar-theme-btn');
  if (!btn) return;
  var isDark = (document.body.getAttribute('data-theme') || 'dark') === 'dark';
  var svg = btn.querySelector('svg');
  if (svg) svg.innerHTML = isDark
    ? '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>'
    : '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>';
};

/* ─── Header clock + system-status pill ───────────── */
ZENTRA._headerTimer = null;
ZENTRA.startHeaderClock = function () {
  if (ZENTRA._headerTimer) return;
  const tick = () => {
    const el = document.getElementById('nav-clock');
    if (el) {
      const d = new Date();
      el.textContent = d.toLocaleTimeString('th-TH', { hour12: false });
    }
    ZENTRA._updateSysPill();
  };
  tick();
  ZENTRA._headerTimer = setInterval(tick, 1000);
};
ZENTRA._updateSysPill = function () {
  const pill = document.getElementById('sys-pill');
  const txt  = document.getElementById('sys-pill-text');
  if (!pill || !txt) return;
  const a = ZENTRA.state.alerts || {};
  const cam = ZENTRA.state.camera;
  let cls = 'ok', label = 'ระบบปกติ';
  if (cam === 'reconnecting' || cam === 'disconnected') { cls = 'warn'; label = 'กล้องไม่พร้อม'; }
  if ((a.emergency || 0) > 0)                            { cls = 'alarm'; label = 'เหตุฉุกเฉิน'; }
  else if ((a.warning || 0) > 0)                          { cls = 'warn';  label = 'มีการแจ้งเตือน'; }
  pill.className = 'sys-pill ' + cls;
  txt.textContent = label;
};

/* ─── Init ─────────────────────────────────────── */
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { ZENTRA.init(); });
} else {
  ZENTRA.init();
}
