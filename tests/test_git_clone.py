"""URL validation + slug derivation for the Git-URL indexing feature.

These are pure/offline checks — no network, no actual `git clone`.
"""
import pytest

from app.services.git_clone import GitCloneError, _slug, _validate


@pytest.mark.parametrize("url", [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo.git",
    "http://gitlab.com/group/sub/repo.git",
    "https://github.com/owner/repo/",
    "git@github.com:owner/repo.git",
    "ssh://git@github.com/owner/repo.git",
])
def test_valid_urls_pass(url):
    assert _validate(url) == url.strip()


@pytest.mark.parametrize("url", [
    "",
    "   ",
    "not a url",
    "ftp://github.com/owner/repo",
    "javascript:alert(1)",
    "/Users/me/local/path",
])
def test_invalid_urls_raise(url):
    with pytest.raises(GitCloneError):
        _validate(url)


def test_slug_is_filesystem_safe_and_stable():
    assert _slug("https://github.com/owner/repo.git") == "github.com_owner_repo"
    assert _slug("git@github.com:owner/repo.git") == "github.com_owner_repo"
    # https and ssh forms of the same repo collapse to one cache dir
    assert _slug("https://github.com/owner/repo") == _slug("git@github.com:owner/repo.git")
