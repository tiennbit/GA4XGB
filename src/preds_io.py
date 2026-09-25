# -*- coding: utf-8 -*-
"""Lưu và đọc dự đoán từng bản ghi (npz) và JSON kết quả, ghi nguyên tử.

Vì sao npz tách khỏi JSON (khung bài 24/9, mục 6): E2, E3, E4, E5, E9 đều là hậu kỳ
trên dự đoán của E1. Lưu dự đoán một lần thì các thí nghiệm sau không phải khớp lại
mô hình, và mọi thí nghiệm đọc CÙNG một ŷ. File ở `~/ga4xgb/preds/<thí nghiệm>/
split<seed>.npz` trên server: ngoài src/ (bị mirror --delete) và ngoài results_cost/
(để pulldir không kéo về Mac).

preds/ KHÔNG BAO GIỜ rời server: không pulldir, không chép sang máy khác hay dịch
vụ nào (AGENTS.md, ràng buộc 1). Không có định danh (tên, ngày sinh, tên trường),
nhưng chỉ số dòng + y + mã trường vẫn là vi dữ liệu nối được với CSV gốc. Để thu hẹp
bề mặt đó, npz chỉ giữ mã trường và mã tỉnh (bootstrap cụm cần ở gần như mọi thí
nghiệm hậu kỳ); năm biến nhóm nhân khẩu của E10 (gender, region, chuyen,
school_size_q, birth_quarter) KHÔNG lưu theo lần chia nữa mà lấy từ
features.load_frame(args.data) ngay trên server: rows(d, part, name, frame=F).

Không dùng pickle (allow_pickle=False khi đọc): file pickle chạy được mã tuỳ ý lúc
nạp, và mảng object không đọc lại được ổn định giữa các bản numpy. Dict mảng được
trải phẳng: `oof={"default": a, "bag10": b}` thành `oof__keys = ["default","bag10"]`,
`oof__0 = a`, `oof__1 = b`. Riêng khoá `meta` (dict thường, JSON được) lưu thành một
chuỗi JSON.

Lược đồ E1 (make_fake_preds.py sinh đúng lược đồ này):
  idx_tr, idx_te        chỉ số dòng gốc (int), thứ tự như splits.outer_split trả về
  y_tr, y_te            y theo đúng thứ tự idx_tr, idx_te
  fold_of               dài n_tr: fold trong mà vị trí đó nằm ở phía giữ lại
  school_code, prov_code
                        dài n (MỌI dòng gốc), đánh chỉ số bằng idx_tr / idx_te;
                        dùng rows(d, "te", "school_code") cho gọn
  oof[center]           dài n_tr: dự đoán OOF (mô hình fold, dự đoán trên fold giữ lại)
  test[center]          dài n_te: dự đoán của mô hình khớp lại trên toàn tập huấn luyện
  test_foldavg[center]  (tuỳ chọn) dài n_te: trung bình dự đoán test của các mô hình
                        fold (chính các mô hình sinh OOF). E5 cần nó cho mục "nguồn ŷ
                        test để áp quy tắc" mà không phải chạy lại E1 (mục 5.3, 0.4).
                        Mỗi mô hình fold dự đoán X_te dựng bằng
                        F.design(fset, train_fold, ..., te): tần suất trường của X_te
                        tính từ hàng huấn luyện CỦA FOLD đó, như X_va.
  trace_oof, trace_test (n_cfg, n_tr), (n_cfg, n_te): vết cấu hình cho E4 (tuỳ chọn)
  trace_test_foldavg    (n_cfg, n_te) (tuỳ chọn): như test_foldavg, cho từng cấu hình
  trace_best_iter       (n_cfg, n_folds): best_iteration của XGBoost, ĐÁNH SỐ TỪ 0
                        (số cây của mô hình fold là best_iteration + 1)
  trace_n_trees         (n_cfg,): số cây đã dùng khi khớp lại trên toàn tập huấn
                        luyện = int(median(trace_best_iter + 1)). E5 nhân hệ số số
                        cây với mảng này, không tự suy lại từ best_iteration.
  trace_oof_rmse        (n_cfg,)
  meta                  dict: seed, feature_set, lo, hi, centers, trace_params,
                        fingerprint, provenance, ...

Tiếp tục sau khi job bị ngắt (sửa theo phản biện hạ tầng): mỗi npz và mỗi JSON
.partial mang `meta.fingerprint` = fingerprint(config), sha256 của cấu hình lượt
chạy cộng code_sha256 của mã đang chạy (provenance). load_or_none và load_partial
từ chối (mặc định: báo lỗi) file có dấu khác, nên một lượt chạy tiếp không trộn các
lần chia tính bằng hai bản mã hay hai cấu hình. Bất biến: mọi mục per_split trong
một .partial có cùng fingerprint với meta.fingerprint của nó, vì file lệch dấu
không bao giờ được đọc tiếp. Script hậu kỳ đưa sha256 của mọi npz đã đọc
(file_sha256) vào config của mình, nên chạy lại E1 sẽ làm hỏng dấu của E2 đang dở.
Vì dấu gồm code_sha256, meta.provenance.code_sha256 của JSON cuối mô tả đúng mã của
MỌI lần chia, kể cả khi lượt chạy được nối qua nhiều lần khởi chạy.

Mẫu dùng trong script thí nghiệm (khác mẫu cũ của center_ablation: .partial giờ có
khoá meta ở gốc, không chỉ là dict per_split):

    fp = preds_io.fingerprint({"feature_set": fs, "n_cfg": n_cfg, "smoke": args.smoke, ...})
    res = preds_io.load_partial(args.out + ".partial", fp, {"per_split": {}})
    for seed in args.seeds:
        if str(seed) in res["per_split"]:
            continue
        p = preds_io.split_path(args.preds_dir, seed)
        d = preds_io.load_or_none(p, fp)          # E1: npz đã có cùng dấu thì bỏ qua fit
        ...
        res["per_split"][str(seed)] = kết_quả
        preds_io.dump_json_atomic(res, args.out + ".partial")
    res["meta"]["provenance"] = provenance.stamp()
    preds_io.dump_json_atomic(res, args.out)

Thay đổi API ngày 24/9 (tương thích ngược, trừ một điểm):
  - split_path(preds_dir, seed, tag=None): tag cho split<seed>_<tag>.npz.
  - rows(d, part, name, frame=None): lấy từ frame khi npz không có mảng đó.
  - Mới: fingerprint, load_or_none, load_partial, file_sha256, to_jsonable,
    FingerprintMismatch. dump_json_atomic và meta của save_split nhận số numpy.
  - KHÔNG tương thích: npz mới không còn gender, region, chuyen, school_size_q,
    birth_quarter. d["gender"] sẽ KeyError; dùng rows(d, part, "gender", frame=F).
    validate_split báo lỗi nếu npz còn chứa các mảng này (npz giả trước 24/9: sinh
    lại bằng tests/make_fake_preds.py).
"""
import copy
import hashlib
import json
import os

import numpy as np

import provenance

META_KEY = "__meta_json__"
E1_REQUIRED = ("idx_tr", "idx_te", "y_tr", "y_te", "school_code", "prov_code", "oof", "test")
CODE_ARRAYS = ("school_code", "prov_code")
GROUP_ARRAYS = ("gender", "region", "chuyen", "school_size_q", "birth_quarter")
# Mảng dài n có thể gặp trong npz (kể cả npz cũ còn biến nhóm): kiểm độ dài.
ROW_ARRAYS = CODE_ARRAYS + GROUP_ARRAYS


class FingerprintMismatch(RuntimeError):
    """File kết quả dở dang được tính bằng mã hoặc cấu hình khác lượt chạy hiện tại."""


def split_path(preds_dir, seed, tag=None):
    """preds_dir/split<seed>.npz, hoặc split<seed>_<tag>.npz khi có tag (ví dụ tập
    đặc trưng hay giai đoạn, để giai đoạn 2 không đọc nhầm file giai đoạn 1)."""
    if tag is None:
        return os.path.join(preds_dir, f"split{int(seed)}.npz")
    tag = str(tag)
    if not tag or os.sep in tag or "/" in tag:
        raise ValueError(f"tag không hợp lệ: {tag!r}")
    return os.path.join(preds_dir, f"split{int(seed)}_{tag}.npz")


def _atomic_target(path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    return path + ".tmp"


# ---------------------------------------------------------------------------
# JSON chịu số numpy
# ---------------------------------------------------------------------------
def to_jsonable(obj):
    """Bản sao ghi được JSON: số numpy thành số Python (kể cả ở KHOÁ dict, nơi
    `default=` của json không với tới), mảng thành list, tuple thành list.

    Vì sao: np.int64 từ np.argmin hay np.median làm json.dump lỗi LÚC GHI, sau khi
    lần chia đã tính xong; chạy tiếp lại tính lại lần chia đó và lại lỗi."""
    if isinstance(obj, dict):
        return {(k.item() if isinstance(k, np.generic) else k): to_jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _np_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"không ghi JSON được kiểu {type(o).__name__}")


def _dumps(obj, **kw):
    return json.dumps(to_jsonable(obj), default=_np_default, ensure_ascii=False, **kw)


# ---------------------------------------------------------------------------
# Dấu lượt chạy cho việc chạy tiếp
# ---------------------------------------------------------------------------
def fingerprint(config, code=None):
    """sha256 (hex) của {config, code_sha256}, JSON chuẩn hoá (khoá sắp xếp).

    config: mọi thứ làm đổi số của MỘT lần chia (tập đặc trưng, giai đoạn, n_cfg,
    max_trees, es_rounds, B, S, lưới, cờ --smoke, sha256 của npz đầu vào). Không
    đưa danh sách seed, --workers, đường dẫn vào: chạy tiếp với thêm seed là hợp lệ.
    code: mặc định provenance.code_hashes() (băm lúc khởi động của mọi module src/
    đang import), nên gọi sau khi đã import xong, ở đầu main()."""
    code = provenance.code_hashes() if code is None else code
    blob = _dumps({"config": config, "code_sha256": code}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _mismatch(path, got, want, on_mismatch):
    msg = (f"{path}: fingerprint {str(got)[:12]} khác lượt chạy hiện tại {want[:12]} "
           "(mã hoặc cấu hình đã đổi). Không trộn lần chia của hai bản mã: xoá/đổi tên "
           "file này để tính lại, hoặc chạy lại đúng mã và cấu hình cũ.")
    if on_mismatch == "recompute":
        print(f"[preds_io] {msg} -> tính lại", flush=True)
        return None
    if on_mismatch != "raise":
        raise ValueError(f"on_mismatch lạ: {on_mismatch}")
    raise FingerprintMismatch(msg)


def load_or_none(path, fingerprint, on_mismatch="raise"):
    """npz đã có và cùng dấu thì trả dict (bỏ qua lần chia này); chưa có thì None.

    Có mà khác dấu (hoặc không có dấu, như npz trước 24/9): mặc định báo
    FingerprintMismatch; on_mismatch="recompute" thì trả None để tính lại và ghi đè."""
    if not os.path.exists(path):
        return None
    d = load_split(path)
    got = (d.get("meta") or {}).get("fingerprint")
    if got == fingerprint:
        return d
    return _mismatch(path, got, fingerprint, on_mismatch)


def file_sha256(path, chunk=1 << 20):
    """sha256 của một file (npz đầu vào): script hậu kỳ ghi nó vào JSON và vào config
    của fingerprint để số của mình nối được về đúng bản dự đoán đã đọc."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# npz dự đoán
# ---------------------------------------------------------------------------
def save_split(path, **arrays):
    """Ghi npz nén, nguyên tử (ghi .tmp rồi os.replace): job bị ngắt giữa chừng không
    để lại file hỏng mang tên thật, nên logic 'đã có file thì bỏ qua' an toàn.
    Đặt meta["fingerprint"] = fingerprint(config) để load_or_none nhận ra file này."""
    flat = {}
    for k, v in arrays.items():
        if "__" in k:
            raise ValueError(f"tên '{k}' chứa '__' (dành cho khoá trải phẳng)")
        if k == "meta":
            flat[META_KEY] = np.array(_dumps(v))
        elif isinstance(v, dict):
            keys = list(v)
            flat[f"{k}__keys"] = np.array(keys, dtype=str)
            for i, kk in enumerate(keys):
                flat[f"{k}__{i}"] = np.asarray(v[kk])
        else:
            flat[k] = np.asarray(v)
    for k, a in flat.items():
        if a.dtype == object:
            raise TypeError(f"'{k}' là mảng object; npz không pickle không lưu được")
    tmp = _atomic_target(path)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **flat)
    os.replace(tmp, path)


def load_split(path):
    """dict: mảng thường, dict mảng dựng lại theo `<tên>__keys`, `meta` là dict.
    Mảng 0 chiều trả về số/chuỗi Python."""
    out, groups = {}, {}
    with np.load(path, allow_pickle=False) as z:
        files = list(z.files)
        for k in files:
            if k == META_KEY:
                out["meta"] = json.loads(str(z[k]))
            elif k.endswith("__keys"):
                groups[k[:-len("__keys")]] = [str(s) for s in z[k]]
        for name, keys in groups.items():
            out[name] = {kk: z[f"{name}__{i}"] for i, kk in enumerate(keys)}
        for k in files:
            if k == META_KEY or "__" in k:
                continue
            a = z[k]
            out[k] = a.item() if a.ndim == 0 else a
    return out


def _frame_array(frame, name):
    if name in ("school_code", "prov_code", "chuyen"):
        return np.asarray(getattr(frame, name))
    if name in getattr(frame, "groups", {}):
        return np.asarray(frame.groups[name])
    raise KeyError(f"Frame không có mảng '{name}'")


def rows(d, part, name, frame=None):
    """Mảng dài n (ROW_ARRAYS) lấy theo phần 'tr' hoặc 'te' của lần chia.

    npz không có mảng đó (biến nhóm E10 không còn lưu theo lần chia) thì lấy từ
    `frame` (features.load_frame(args.data), trên server). Khi dùng frame, kiểm y của
    frame khớp y trong npz: nhầm file dữ liệu sẽ báo lỗi thay vì lặng lẽ lệch dòng."""
    idx = np.asarray(d["idx_" + part])
    if name in d:
        return np.asarray(d[name])[idx]
    if frame is None:
        raise KeyError(f"'{name}' không lưu trong npz (từ 24/9 biến nhóm E10 lấy từ CSV "
                       "trên server): gọi rows(d, part, name, frame=features.load_frame(data))")
    if not np.array_equal(np.asarray(frame.y, dtype=float)[idx], np.asarray(d["y_" + part], dtype=float)):
        raise ValueError("y của frame không khớp y trong npz: frame dựng từ file dữ liệu khác")
    return _frame_array(frame, name)[idx]


def _check_matrix(d, key, n_cfg, n, err):
    a = np.asarray(d[key])
    if a.ndim != 2 or a.shape[1] != n or (n_cfg is not None and a.shape[0] != n_cfg):
        err.append(f"{key} có dạng {a.shape}, cần ({n_cfg if n_cfg is not None else 'n_cfg'}, {n})")
    elif not np.all(np.isfinite(a)):
        bad = np.where(~np.isfinite(a).all(axis=1))[0]
        err.append(f"{key} có NaN/inf ở cấu hình {bad[:10].tolist()}")


def validate_split(d, required=E1_REQUIRED):
    """Kiểm lược đồ, độ dài và giá trị hữu hạn; trả danh sách lỗi (rỗng là hợp lệ)."""
    err = [f"thiếu '{k}'" for k in required if k not in d]
    if err:
        return err
    n_tr, n_te = len(d["idx_tr"]), len(d["idx_te"])
    if len(d["y_tr"]) != n_tr or len(d["y_te"]) != n_te:
        err.append("y_tr/y_te lệch độ dài với idx_tr/idx_te")
    if np.intersect1d(d["idx_tr"], d["idx_te"]).size:
        err.append("idx_tr và idx_te giao nhau")
    n_min = int(max(np.max(d["idx_tr"]), np.max(d["idx_te"]))) + 1
    for k in ROW_ARRAYS:
        if k in d and len(d[k]) < n_min:
            err.append(f"'{k}' phải dài n (mọi dòng gốc), đang dài {len(d[k])}")
    leaked = [k for k in GROUP_ARRAYS if k in d]
    if leaked:
        err.append(f"npz chứa biến nhóm nhân khẩu {leaked}: không lưu theo lần chia, E10 lấy "
                   "từ features.load_frame qua rows(..., frame=F)")
    if "fold_of" in d:
        fo = np.asarray(d["fold_of"])
        if len(fo) != n_tr:
            err.append(f"fold_of dài {len(fo)}, cần n_tr = {n_tr}")
        elif fo.size and fo.min() < 0:
            err.append("fold_of có vị trí không thuộc fold giữ lại nào (-1)")
    for part, n in (("oof", n_tr), ("test", n_te), ("test_foldavg", n_te)):
        for c, a in d.get(part, {}).items():
            if len(a) != n:
                err.append(f"{part}[{c}] dài {len(a)}, cần {n}")
            if not np.all(np.isfinite(a)):
                err.append(f"{part}[{c}] có NaN/inf")
    extra = set(d.get("test_foldavg", {})) - set(d.get("test", {}))
    if extra:
        err.append(f"test_foldavg có trung tâm không có trong test: {sorted(extra)}")
    n_cfg = None
    for key, n in (("trace_oof", n_tr), ("trace_test", n_te), ("trace_test_foldavg", n_te)):
        if key in d:
            _check_matrix(d, key, n_cfg, n, err)
            n_cfg = np.asarray(d[key]).shape[0] if n_cfg is None else n_cfg
    if "trace_best_iter" in d:
        bi = np.asarray(d["trace_best_iter"])
        if bi.ndim != 2 or (n_cfg is not None and bi.shape[0] != n_cfg):
            err.append(f"trace_best_iter có dạng {bi.shape}, cần (n_cfg, n_folds)")
        elif bi.size and bi.min() < 0:
            err.append("trace_best_iter âm")
    for key in ("trace_n_trees", "trace_oof_rmse"):
        if key in d:
            a = np.asarray(d[key])
            if a.ndim != 1 or (n_cfg is not None and len(a) != n_cfg):
                err.append(f"{key} có dạng {a.shape}, cần ({n_cfg},)")
            elif not np.all(np.isfinite(a)):
                err.append(f"{key} có NaN/inf")
    if "trace_n_trees" in d and "trace_best_iter" in d and not err:
        want = np.median(np.asarray(d["trace_best_iter"]) + 1, axis=1).astype(int)
        if not np.array_equal(np.asarray(d["trace_n_trees"]).astype(int), want):
            err.append("trace_n_trees khác int(median(trace_best_iter + 1))")
    return err


# ---------------------------------------------------------------------------
# JSON kết quả: ghi nguyên tử, tiếp tục theo lần chia
# ---------------------------------------------------------------------------
def dump_json_atomic(obj, path, indent=1):
    """Ghi JSON nguyên tử (.tmp rồi os.replace); số và mảng numpy được chuyển sẵn."""
    tmp = _atomic_target(path)
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(_dumps(obj, indent=indent))
    os.replace(tmp, path)


def load_json(path, default=None):
    """Đọc JSON; không có file thì trả default. File hỏng thì báo lỗi, KHÔNG nuốt:
    lặng lẽ bắt đầu lại từ đầu sẽ xoá mất các lần chia đã xong."""
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_partial(path, fingerprint, default, on_mismatch="raise"):
    """Đọc JSON dở dang (<out>.partial) để chạy tiếp, chỉ khi cùng dấu lượt chạy.

    Chưa có file: trả bản sao của `default` với meta.fingerprint đã điền. Có và cùng
    meta.fingerprint: trả nguyên file (các lần chia đã xong được bỏ qua). Khác dấu
    hoặc không có dấu: mặc định FingerprintMismatch; on_mismatch="recompute" thì bỏ
    file cũ, bắt đầu lại từ `default`."""
    fresh = copy.deepcopy(default)
    fresh.setdefault("meta", {})["fingerprint"] = fingerprint
    obj = load_json(path)
    if obj is None:
        return fresh
    got = (obj.get("meta") or {}).get("fingerprint")
    if got == fingerprint:
        return obj
    _mismatch(path, got, fingerprint, on_mismatch)      # báo lỗi, hoặc cho tính lại
    return fresh
