#!/usr/bin/env python3
"""X11 screen and window capture for Linux.

The game runs through Proton on Linux, but its pixels are ordinary X11 desktop
pixels once composed.  Pillow uses XGetImage for the actual copy; python-xlib is
used only for window enumeration, geometry, and the active-window guard.

This backend deliberately targets X11.  Wayland does not allow an application
to capture arbitrary windows without going through the desktop portal and an
interactive chooser.  KDE can run either session type; this module reports a
clear error when invoked from Wayland instead of silently returning black.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .base import Capture, CaptureError, Window, app_matches


def _display():
    if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" and not os.environ.get("DISPLAY"):
        raise CaptureError(
            "Linux capture currently requires an X11 session. Log out, choose "
            "Plasma (X11), and log in again.")
    try:
        from Xlib import display
        return display.Display()
    except Exception as exc:  # noqa: BLE001
        raise CaptureError(
            f"cannot connect to the X11 display {os.environ.get('DISPLAY', '(unset)')!r}: {exc}") from exc


def _text_property(win, atom_name: str) -> str:
    try:
        d = win.display
        prop = win.get_full_property(d.intern_atom(atom_name), 0)
        if prop is None or prop.value is None:
            return ""
        value = prop.value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").rstrip("\0")
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


def _title(win) -> str:
    return _text_property(win, "_NET_WM_NAME") or (win.get_wm_name() or "")


def _app(win) -> str:
    try:
        cls = win.get_wm_class() or ()
        app = cls[-1] if cls else ""
        # Proton exposes games as an opaque Steam id (for example
        # ``steam_app_1003400``), not as the Windows executable name.  Include
        # the title for those windows so the normal Journeys/JiME hint and the
        # foreground guard can identify the game.  Restricting this to Proton's
        # WM_CLASS avoids matching a terminal merely because its title contains
        # this repository's name.
        if app.lower().startswith("steam_app_"):
            title = _title(win)
            return f"{app} {title}" if title else app
        return app
    except Exception:  # noqa: BLE001
        return ""


def _matches(app: str, hint: str) -> bool:
    # Proton normally exposes JiME.exe as WM_CLASS, while the native builds use
    # "Journeys in Middle-earth".  Treat those as the same documented hint.
    if app_matches(app, hint):
        return True
    return hint.strip().lower() == "journeys" and "jime" in app.lower()


def _client_windows(d) -> list:
    root = d.screen().root
    prop = root.get_full_property(d.intern_atom("_NET_CLIENT_LIST"), 0)
    if prop is not None:
        return [d.create_resource_object("window", int(w)) for w in prop.value]
    return list(root.query_tree().children)


def list_windows(app_hint: str = "") -> list[Window]:
    d = _display()
    try:
        root = d.screen().root
        out: list[Window] = []
        for win in _client_windows(d):
            try:
                attrs = win.get_attributes()
                geom = win.get_geometry()
                if attrs.map_state != 2 or geom.width < 1 or geom.height < 1:
                    continue
                pos = win.translate_coords(root, 0, 0)
                title, app = _title(win), _app(win)
                if not title or not _matches(app, app_hint):
                    continue
                out.append(Window(handle=(int(win.id), pos.x, pos.y), title=title,
                                  app=app, width=geom.width, height=geom.height))
            except Exception:  # noqa: BLE001
                continue
        return out
    finally:
        d.close()


def foreground() -> str:
    d = _display()
    try:
        root = d.screen().root
        prop = root.get_full_property(d.intern_atom("_NET_ACTIVE_WINDOW"), 0)
        if prop is None or not len(prop.value):
            return ""
        return _app(d.create_resource_object("window", int(prop.value[0])))
    finally:
        d.close()


@dataclass
class X11Capture(Capture):
    bbox: tuple[int, int, int, int] | None = None
    window_id: int | None = None

    def grab(self) -> np.ndarray | None:
        from PIL import ImageGrab
        bbox = self.bbox
        if self.window_id is not None:
            # Refresh geometry because a window can move or resize after it was
            # selected. Return None if it has closed.
            d = _display()
            try:
                win = d.create_resource_object("window", self.window_id)
                geom = win.get_geometry()
                pos = win.translate_coords(d.screen().root, 0, 0)
                bbox = (pos.x, pos.y, pos.x + geom.width, pos.y + geom.height)
            except Exception:  # noqa: BLE001
                return None
            finally:
                d.close()
        try:
            return np.asarray(ImageGrab.grab(bbox=bbox).convert("L"))
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"X11 screen capture failed: {exc}") from exc

    def close(self) -> None:
        pass


def open_display(index: int = 0) -> Capture:
    if index != 0:
        raise CaptureError(
            "the Linux X11 backend exposes the combined desktop as display 0")
    # Connect once now so configuration errors appear before the play loop.
    d = _display()
    d.close()
    return X11Capture()


def open_window(title_hint: str = "", app_hint: str = "Journeys",
                wait: float = 0.0) -> Capture:
    deadline = time.monotonic() + wait
    announced = False
    while True:
        matches = [w for w in list_windows("")
                   if (not title_hint or title_hint.lower() in w.title.lower())
                   and _matches(w.app, app_hint)
                   and w.width > 200 and w.height > 200]
        if matches:
            best = max(matches, key=lambda w: w.width * w.height)
            wid, x, y = best.handle
            cap = X11Capture((x, y, x + best.width, y + best.height), wid)
            cap.window = best  # type: ignore[attr-defined]
            return cap
        if time.monotonic() >= deadline:
            break
        if not announced:
            print("waiting for the game window — start it now", flush=True)
            announced = True
        time.sleep(0.5)
    seen = list_windows("")
    raise CaptureError(
        f"no window matching app={app_hint!r} title={title_hint!r}.\n\n"
        "Visible windows:\n  " + "\n  ".join(str(w) for w in seen[:12]))
