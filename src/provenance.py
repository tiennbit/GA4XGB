# -*- coding: utf-8 -*-
"""meta.provenance cho mọi JSON kết quả: số nào ra từ bản code nào, trên máy nào.

Vì sao (khung bài 24/9, mục 0.1 và 6): `results_cost/CODE.stamp` dùng chung bị ghi
đè ở mỗi lần khởi chạy, nên commit của các lượt thăm dò trước chỉ còn suy từ thông
điệp commit. Từ E0, mỗi JSON tự mang dấu của mình:

    {commit, dirty, diff_sha256, untracked_sha256, lock_sha256, code_sha256 {file: sha},
     launched, stamped, host, argv, lib_versions, stamp_source,
     code_matches_stamp, stamp_mismatch, code_changed_since_start, code_changed_files}

ĐỊNH DANH CÓ THẨM QUYỀN LÀ code_sha256, không phải commit. Server không có git
(AGENTS.md: sửa ở local, chạy ở server), nên commit, dirty, diff chỉ là thông tin
đi kèm; code_sha256 đối chiếu thẳng được với `sync_server.local.sh verify` (cùng
sha256) và với `src_sha256` của stamp.

Nguồn của commit, dirty, diff:
  1. Nếu biến môi trường RUN_STAMP trỏ tới một file JSON (do script đồng bộ tạo ở
     Mac lúc `sh`, xem `python3 src/provenance.py --emit ...`), lấy từ file đó. File
     stamp là riêng cho từng lần khởi chạy và không bị ghi lại, nên đọc lúc gọi
     stamp() là đủ.
  2. Nếu không, thử git tại gốc repo (thư mục cha của src/).
  3. Không được thì ghi "unknown". Không bao giờ dừng job chỉ vì thiếu dấu.

Ảnh chụp lúc khởi động (sửa theo phản biện hạ tầng): `sync_server.local.sh code`
mirror src/ với --delete trong khi job dài đang chạy (ví dụ đẩy mã E3 giữa lượt
E2b 10,5 giờ). Nếu băm file trên đĩa lúc ghi JSON thì dấu sẽ nói về mã job chưa
từng chạy. Vì vậy, lúc module này được import (đầu tiến trình, cùng đợt import với
các module khác), nó băm MỌI file src/*.py một lần và nhớ giờ khởi động:
  - code_sha256 lấy băm lúc khởi động cho các module đã import;
  - launched là giờ khởi động, stamped là giờ gọi stamp();
  - code_changed_since_start = True khi file trên đĩa đã khác băm lúc khởi động
    (liệt kê trong code_changed_files); khi đó số liệu vẫn đúng với mã đã chạy,
    nhưng cây src/ hiện tại không còn là mã đó.
Giả định: script import mọi module src/ ở đầu file (quy ước của repo). Module import
muộn bên trong hàm sau khi đĩa đã đổi sẽ bị ghi băm lúc khởi động, nhưng cờ
code_changed_since_start vẫn bật nên không lặng lẽ sai.

Đối chiếu với stamp: stamp do emit_run_stamp ghi có src_sha256 (mọi file src/*.py ở
Mac lúc `sh`). stamp() so từng file trong code_sha256 với nó; lệch thì
code_matches_stamp = False, stamp_mismatch liệt kê file lệch và in cảnh báo. Đây
là lỗi mà bước `verify` bắt buộc sinh ra để chặn: chạy `sh` với stamp cũ, hoặc sửa
ở Mac mà chưa đẩy lại. Stamp dạng dòng cũ không có src_sha256 thì
code_matches_stamp = None (không biết).

Phạm vi `dirty`, `diff_sha256` và `untracked_sha256` là thư mục src/ (khớp cờ dirty
của script đồng bộ): diff của file hình hay JSON kết quả không nói gì về code sinh
ra số. diff_sha256 là sha256 của `git diff HEAD -- src`, rỗng khi không có thay đổi
trên file đã theo dõi. `git diff` không thấy file CHƯA theo dõi, nên hai bản khác
nhau của một file mới cho cùng (commit, dirty, diff_sha256); untracked_sha256
{file: sha} bổ sung phần đó cho tới khi các file mới được commit (E0 bước 5).
"""
import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
LOCK_FILE = REPO_DIR / "requirements.lock.txt"
LIBS = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "joblib", "matplotlib"]
UNKNOWN = "unknown"


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p):
    try:
        return _sha256_bytes(Path(p).read_bytes())
    except OSError:
        return "missing"


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _disk_hashes():
    return {p.name: _sha256_file(p) for p in sorted(SRC_DIR.glob("*.py"))}


# Ảnh chụp lúc khởi động: xem docstring. Băm khoảng 50 file nhỏ, vài mili giây.
_BOOT_TIME = _now()
_BOOT_DISK = _disk_hashes()
_WARNED = set()


def _warn(msg):
    """In cảnh báo ra stderr, mỗi nội dung một lần (stamp() có thể được gọi mỗi lần chia)."""
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[provenance] CẢNH BÁO: {msg}", file=sys.stderr, flush=True)


def lib_versions():
    """Phiên bản cài đặt (đọc metadata, không import thư viện nặng)."""
    out = {"python": platform.python_version()}
    for name in LIBS:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def print_versions(file=None):
    """In phiên bản thư viện ở đầu lượt chạy: lệch bản là lệch số (AGENTS.md, ràng buộc 2)."""
    v = lib_versions()
    print("versions: " + " ".join(f"{k}={x}" for k, x in v.items()), file=file or sys.stdout,
          flush=True)
    return v


def _imported_src_files():
    """{tên file: Path} của mọi module .py trong src/ đang được import, kể cả __main__."""
    out = {}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f or not f.endswith(".py"):
            continue
        p = Path(f).resolve()
        if p.parent == SRC_DIR:
            out[p.name] = p
    return out


def code_hashes():
    """{tên file: sha256} cho mọi module src/ đang import, theo ẢNH CHỤP LÚC KHỞI ĐỘNG.

    File chưa có lúc khởi động (tạo ra rồi mới import) thì băm tại chỗ."""
    out = {}
    for name, p in _imported_src_files().items():
        out[name] = _BOOT_DISK.get(name) or _sha256_file(p)
    return dict(sorted(out.items()))


def code_changes():
    """Tên các module src/ đang import mà file trên đĩa đã khác băm lúc khởi động."""
    boot = code_hashes()
    return sorted(n for n, p in _imported_src_files().items() if _sha256_file(p) != boot[n])


def _git(*args):
    r = subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace"))
    return r.stdout


def git_state():
    """{commit, dirty, diff_sha256, untracked_sha256} từ git, phạm vi src/.

    Mọi lỗi (không có git, không phải repo) cho unknown."""
    try:
        commit = _git("rev-parse", "HEAD").decode().strip()
        dirty = bool(_git("status", "--porcelain", "--", "src").strip())
        diff = _git("diff", "HEAD", "--", "src")
        others = _git("ls-files", "--others", "--exclude-standard", "-z", "--", "src")
        untracked = {}
        for rel in sorted(s for s in others.decode("utf-8", "replace").split("\0") if s):
            if rel.endswith(".py"):
                untracked[rel] = _sha256_file(REPO_DIR / rel)
        return {"commit": commit, "dirty": dirty,
                "diff_sha256": _sha256_bytes(diff) if diff else "",
                "untracked_sha256": untracked}
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {"commit": UNKNOWN, "dirty": UNKNOWN, "diff_sha256": UNKNOWN,
                "untracked_sha256": UNKNOWN}


def _read_run_stamp(path):
    """Đọc stamp JSON. Chấp nhận cả dạng dòng cũ 'commit=abc dirty=yes launched=...'
    của CODE.stamp để lượt chạy dở dang vẫn có dấu."""
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        d = dict(tok.split("=", 1) for tok in text.split() if "=" in tok)
        d["dirty"] = d.get("dirty", "no") == "yes"
    return d


def check_against_stamp(code, run_stamp):
    """(khớp?, [file lệch]) giữa code_sha256 và src_sha256 của stamp.

    khớp là None khi stamp không có src_sha256 (dạng dòng cũ): không biết. Module có
    trong code mà không có trong stamp cũng tính là lệch (file mới chưa đẩy qua stamp)."""
    src = run_stamp.get("src_sha256") if isinstance(run_stamp, dict) else None
    if not isinstance(src, dict):
        return None, []
    bad = sorted(f for f, h in code.items() if src.get(f) != h)
    return (not bad), bad


def stamp(extra=None):
    """meta.provenance cho JSON kết quả. `extra` (dict) được gộp vào cuối."""
    src = "git"
    rs_path = os.environ.get("RUN_STAMP")
    run_stamp = None
    if rs_path:
        try:
            run_stamp = _read_run_stamp(rs_path)
            base = {k: run_stamp.get(k, UNKNOWN)
                    for k in ("commit", "dirty", "diff_sha256", "untracked_sha256")}
            src = f"RUN_STAMP:{rs_path}"
        except (OSError, ValueError) as exc:
            _warn(f"không đọc được RUN_STAMP={rs_path}: {exc}; thử git")
            base = git_state()
            src = "git (RUN_STAMP lỗi)"
    else:
        base = git_state()
    if base.get("commit") == UNKNOWN and src == "git":
        src = UNKNOWN
    code = code_hashes()
    matches, mismatch = (None, []) if run_stamp is None else check_against_stamp(code, run_stamp)
    if matches is False:
        _warn(f"mã đang chạy KHÔNG khớp stamp {rs_path} ở {mismatch}: commit/diff trong "
              "stamp không mô tả mã này (stamp cũ, hoặc sửa ở Mac mà chưa đẩy và verify)")
    changed = code_changes()
    if changed:
        _warn(f"file src/ đã đổi trên đĩa kể từ lúc khởi động: {changed}; code_sha256 ghi "
              "băm lúc khởi động (mã đã chạy), không phải cây hiện tại")
    out = base | {
        "lock_sha256": _sha256_file(LOCK_FILE),
        "code_sha256": code,
        "launched": _BOOT_TIME,
        "stamped": _now(),
        "host": socket.gethostname(),
        "argv": list(sys.argv),
        "lib_versions": lib_versions(),
        "stamp_source": src,
        "code_matches_stamp": matches,
        "stamp_mismatch": mismatch,
        "code_changed_since_start": bool(changed),
        "code_changed_files": changed,
    }
    if run_stamp is not None:
        out["run_stamp"] = run_stamp
    if extra:
        out |= dict(extra)
    return out


def emit_run_stamp(path, label=None):
    """Ghi stamp JSON ở Mac để đẩy lên server làm RUN_STAMP (server không có git)."""
    d = git_state() | {
        "launched": _now(), "host": socket.gethostname(), "lock_sha256": _sha256_file(LOCK_FILE),
        "src_sha256": _disk_hashes()}
    if label:
        d["label"] = label
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="In provenance hiện tại, hoặc ghi stamp cho RUN_STAMP")
    ap.add_argument("--emit", help="đường dẫn file stamp JSON cần ghi (chạy ở Mac)")
    ap.add_argument("--label", help="tên lượt chạy ghi vào stamp")
    a = ap.parse_args()
    if a.emit:
        d = emit_run_stamp(a.emit, a.label)
        print(f"commit={d['commit'][:12]} dirty={d['dirty']} -> {a.emit}")
    else:
        print(json.dumps(stamp(), indent=1, ensure_ascii=False))
