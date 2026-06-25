"""Clone a remote git repo into a local cache dir for indexing.

This powers the "index from a Git URL" feature, which sits *alongside* the
existing local-path indexing — it does not replace it. We shallow-clone
(`--depth 1`) into ``<data_dir>/repos/<slug>`` and hand that path to the
normal ``full_index`` pipeline, so chunking, retrieval, and GitHub citations
all work exactly as they do for a local repo (the cloned checkout's
``remote.origin.url`` lets ``Indexer._git_meta`` recover owner/repo/commit).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from app.config import settings

# Accept the common remote forms and nothing exotic. We pass args as a list
# (no shell), so this is fail-fast hygiene rather than the only defense.
_HTTPS_RE = re.compile(r"^https?://[\w.\-]+(?::\d+)?/[\w.\-/~%]+?(?:\.git)?/?$")
_SSH_RE = re.compile(r"^(?:ssh://)?git@[\w.\-]+[:/][\w.\-/~]+?(?:\.git)?/?$")


class GitCloneError(ValueError):
    """Raised when a URL is invalid or the clone fails."""


def _validate(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise GitCloneError("No git URL provided.")
    if not (_HTTPS_RE.match(url) or _SSH_RE.match(url)):
        raise GitCloneError(f"Not a valid git URL: {url!r}")
    return url


def _slug(url: str) -> str:
    """Stable, filesystem-safe folder name derived from host/owner/repo."""
    s = re.sub(r"^https?://", "", url)
    s = re.sub(r"^(?:ssh://)?git@", "", s)
    s = s.replace(":", "/")
    s = re.sub(r"\.git/?$", "", s).strip("/")
    s = re.sub(r"[^\w.\-]+", "_", s)
    return s or "repo"


def clone_repo(git_url: str, branch: str | None = None) -> str:
    """Shallow-clone ``git_url`` into the repos cache; return the local path.

    Any previous copy at the same slug is removed first, so a re-index always
    reflects the current remote HEAD. ``GIT_TERMINAL_PROMPT=0`` makes private
    repos fail fast instead of hanging on a credential prompt.
    """
    url = _validate(git_url)
    dest = settings.repos_cache_dir / _slug(url)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", "--depth", "1", "--single-branch"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dest)]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=settings.git_clone_timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as e:  # git not installed
        raise GitCloneError("git is not installed or not on PATH.") from e
    except subprocess.TimeoutExpired:
        shutil.rmtree(dest, ignore_errors=True)
        raise GitCloneError(
            f"Clone timed out after {settings.git_clone_timeout}s."
        )

    if proc.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        detail = (proc.stderr or proc.stdout or "git clone failed").strip().splitlines()
        raise GitCloneError(detail[-1] if detail else "git clone failed")

    return str(dest)
