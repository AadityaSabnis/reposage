"""Local file watcher (dev only).

Watches REPO_PATH and feeds changes into the same incremental_update used
by the /webhook/git-push endpoint, with a short debounce so a burst of
saves collapses into one update.

Run standalone (server should NOT also be writing the same index):
    python -m app.watcher
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Set

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.chunking.registry import should_index
from app.config import settings
from app.deps import get_indexer

log = logging.getLogger("reposage.watcher")

DEBOUNCE_SEC = 1.5


class _Handler(FileSystemEventHandler):
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()
        self._changed: Set[str] = set()
        self._deleted: Set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _track(self, path: str, deleted: bool) -> None:
        p = Path(path)
        # On delete the file is gone, so we can't stat it; index by path only.
        if not deleted and not should_index(p):
            return
        if deleted and p.suffix.lower() not in _watchable_suffixes():
            return
        with self._lock:
            if deleted:
                self._deleted.add(str(p))
                self._changed.discard(str(p))
            else:
                self._changed.add(str(p))
                self._deleted.discard(str(p))
            self._schedule()

    def _schedule(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE_SEC, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            changed = sorted(self._changed)
            deleted = sorted(self._deleted)
            self._changed.clear()
            self._deleted.clear()
        if not changed and not deleted:
            return
        log.info("Incremental update: %d changed, %d deleted", len(changed), len(deleted))
        stats = get_indexer().incremental_update(changed, deleted)
        log.info("Done: %s", stats)

    # watchdog callbacks
    def on_modified(self, event):
        if not event.is_directory:
            self._track(event.src_path, deleted=False)

    def on_created(self, event):
        if not event.is_directory:
            self._track(event.src_path, deleted=False)

    def on_deleted(self, event):
        if not event.is_directory:
            self._track(event.src_path, deleted=True)

    def on_moved(self, event):
        if not event.is_directory:
            self._track(event.src_path, deleted=True)
            self._track(event.dest_path, deleted=False)


def _watchable_suffixes() -> Set[str]:
    from app.chunking.registry import EXT_TO_LANGUAGE, FALLBACK_TEXT_EXTS
    return set(EXT_TO_LANGUAGE) | set(FALLBACK_TEXT_EXTS)


def watch(repo_path: str | None = None) -> None:
    repo_path = str(Path(repo_path or settings.repo_path).resolve())
    handler = _Handler(repo_path)
    observer = Observer()
    observer.schedule(handler, repo_path, recursive=True)
    observer.start()
    log.info("Watching %s for changes (Ctrl-C to stop)…", repo_path)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    watch()
