# -*- coding: utf-8 -*-
"""GA tối ưu hyperparameter XGBoost cho bài toán dự đoán Điểm HSA (hồi quy).

Thiết kế theo khung GA4RF (IEEE Access 2025) nhưng:
- Mã hoá THỰC (real-coded) thay vì nhị phân — hyperparameter XGBoost phần lớn liên tục.
- Fitness chọn được qua --metric: rmse | mae | r2 (r2 chuyển thành minimize 1-R2).
- Tournament selection + blend crossover (BLX-alpha) + gaussian mutation + elitism.
- Mỗi lần đánh giá đều ghi lại CẢ 3 metric (rmse, mae, r2) để phân tích chéo.

Chromosome (7 gene):
  n_estimators [100, 800] (int)     learning_rate [0.01, 0.30] (log)
  max_depth    [3, 12]    (int)     subsample     [0.5, 1.0]
  colsample_bytree [0.5, 1.0]       min_child_weight [1, 20] (int)
  reg_lambda   [0.0, 10.0]

Chạy:  python3 src/ga_xgb.py --metric rmse [--n-jobs -1] [--seed 42]
Kết quả: results/ga_{metric}_log.jsonl (mỗi thế hệ 1 dòng), results/ga_{metric}_best.json
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_and_preprocess

# Bin điểm cho tail-weighted fitness (khớp với phân tích sai số theo bin)
TAIL_BINS = [0, 50, 60, 70, 80, 90, 100, 110, 150]

POP_SIZE = 24
N_GENERATIONS = 30
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.15      # xác suất đột biến mỗi gene
TOURNAMENT_K = 3
ELITISM = 2
PATIENCE = 10             # dừng sớm nếu không cải thiện
CV_FOLDS = 3
BLX_ALPHA = 0.3

# (tên, min, max, kiểu, thang-log?)
BASE_GENES = [
    ("n_estimators",     100, 800,  int,   False),
    ("max_depth",        3,   12,   int,   False),
    ("learning_rate",    0.01, 0.30, float, True),
    ("subsample",        0.5, 1.0,  float, False),
    ("colsample_bytree", 0.5, 1.0,  float, False),
    ("min_child_weight", 1,   20,   int,   False),
    ("reg_lambda",       0.0, 10.0, float, False),
]

# Gene thứ 8 (tuỳ chọn, bật bằng --loss-weight): cường độ trọng số mẫu TRONG LOSS.
# w_i ∝ n_{b(i)}^(-beta), chuẩn hoá trung bình 1.
#   beta=0 -> mọi mẫu trọng số 1 (hành vi XGBoost thường)
#   beta=1 -> trọng số macro trong loss (bin thưa được đền bù hoàn toàn)
# Đối xứng có chủ ý với alpha của fitness: alpha điều khiển ĐÁNH GIÁ, beta điều khiển HỌC.
# GA tự tìm beta -> hợp nhất can thiệp mức fitness và mức loss trong một vòng tối ưu.
WEIGHT_GENE = ("weight_beta", 0.0, 1.0, float, False)

GENES = list(BASE_GENES)   # bị ghi đè trong GA.__init__ nếu bật --loss-weight


def decode(ind, genes=None):
    params = {}
    for (name, lo, hi, typ, _), v in zip(genes or GENES, ind):
        v = min(max(v, lo), hi)
        params[name] = int(round(v)) if typ is int else float(v)
    return params


def split_params(params):
    """Tách gene weight_beta (tham số của TA) khỏi hyperparameter của XGBoost."""
    p = dict(params)
    beta = p.pop("weight_beta", None)
    return p, beta


def sample_weights(y_tr, beta):
    """Trọng số mẫu theo bin: w_i ∝ n_b^(-beta), chuẩn hoá trung bình 1.

    Tính CHỈ từ fold huấn luyện -> không rò rỉ thông tin từ fold kiểm định."""
    b = np.digitize(y_tr, TAIL_BINS[1:-1])
    cnt = pd.Series(b).value_counts()
    w = np.asarray([cnt[bi] ** (-beta) for bi in b], dtype=float)
    return w / w.mean()


def key_of(ind):
    return tuple(sorted(decode(ind).items()))


class GA:
    def __init__(self, metric, seed, n_jobs, alpha=1.0, loss_weight=False):
        global GENES
        self.metric = metric
        self.seed = seed
        self.n_jobs = n_jobs
        self.alpha = alpha    # mức nhấn đuôi cho metric 'tail': 0=RMSE thường, 1=macro-RMSE
        self.loss_weight = loss_weight
        self.genes = list(BASE_GENES) + ([WEIGHT_GENE] if loss_weight else [])
        GENES = self.genes    # decode()/key_of() dùng biến module-level
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cache = {}       # key -> {"rmse":..., "mae":..., "r2":...}
        self.n_evals = 0

        X, y = load_and_preprocess()
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(
            X, y, test_size=0.2, random_state=42)   # split cố định, độc lập với seed GA
        self.kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # ---- fitness ----
    def tail_rmse(self, yv, pred):
        """RMSE có trọng số theo bin điểm: w_b ∝ n_b^(1-alpha), chuẩn hoá tổng 1.

        alpha=0 -> w_b ∝ n_b: RMSE thường (micro, theo mật độ);
        alpha=1 -> w_b bằng nhau: macro-RMSE, bin đuôi thưa được nhấn tương ứng."""
        bins = np.digitize(yv, TAIL_BINS[1:-1])   # chỉ số bin cho từng mẫu
        se = (np.asarray(yv, dtype=float) - np.asarray(pred, dtype=float)) ** 2
        mses, ws = [], []
        for b in np.unique(bins):
            mask = bins == b
            mses.append(se[mask].mean())
            ws.append(mask.sum() ** (1.0 - self.alpha))
        ws = np.asarray(ws) / np.sum(ws)
        return float(np.sqrt(np.sum(ws * np.asarray(mses))))

    def scores_of(self, ind):
        k = key_of(ind)
        if k in self.cache:
            return self.cache[k]
        xgb_params, beta = split_params(decode(ind, self.genes))
        rmses, maes, r2s, tails = [], [], [], []
        for tr, va in self.kf.split(self.X_tr):
            m = XGBRegressor(tree_method="hist", random_state=42,
                             n_jobs=self.n_jobs, verbosity=0, **xgb_params)
            ytr_fold = self.y_tr.iloc[tr]
            sw = sample_weights(ytr_fold, beta) if beta is not None else None
            m.fit(self.X_tr.iloc[tr], ytr_fold, sample_weight=sw)
            pred = m.predict(self.X_tr.iloc[va])
            yv = self.y_tr.iloc[va]
            rmses.append(float(np.sqrt(mean_squared_error(yv, pred))))
            maes.append(float(mean_absolute_error(yv, pred)))
            r2s.append(float(r2_score(yv, pred)))
            tails.append(self.tail_rmse(yv, pred))
        s = {"rmse": float(np.mean(rmses)), "mae": float(np.mean(maes)),
             "r2": float(np.mean(r2s)), "tail": float(np.mean(tails))}
        self.cache[k] = s
        self.n_evals += 1
        return s

    def fitness(self, ind):
        s = self.scores_of(ind)
        if self.metric == "r2":
            return 1.0 - s["r2"]        # maximize R2 <=> minimize 1-R2
        return s[self.metric]           # rmse / mae / tail: minimize trực tiếp

    # ---- toán tử GA ----
    def random_individual(self):
        ind = []
        for name, lo, hi, typ, log in self.genes:
            if log:
                v = float(np.exp(self.np_rng.uniform(np.log(lo), np.log(hi))))
            else:
                v = float(self.np_rng.uniform(lo, hi))
            ind.append(v)
        return ind

    def tournament(self, pop, fits):
        best = None
        for _ in range(TOURNAMENT_K):
            i = self.rng.randrange(len(pop))
            if best is None or fits[i] < fits[best]:
                best = i
        return pop[best][:]

    def crossover(self, p1, p2):
        """BLX-alpha: con lấy trong khoảng mở rộng giữa 2 cha mẹ."""
        c1, c2 = p1[:], p2[:]
        if self.rng.random() < CROSSOVER_RATE:
            for g in range(len(self.genes)):
                lo_v, hi_v = min(p1[g], p2[g]), max(p1[g], p2[g])
                span = hi_v - lo_v
                a, b = lo_v - BLX_ALPHA * span, hi_v + BLX_ALPHA * span
                c1[g] = float(self.np_rng.uniform(a, b))
                c2[g] = float(self.np_rng.uniform(a, b))
        return c1, c2

    def mutate(self, ind):
        for g, (name, lo, hi, typ, log) in enumerate(self.genes):
            if self.rng.random() < MUTATION_RATE:
                sigma = (hi - lo) * 0.15
                ind[g] = float(np.clip(ind[g] + self.np_rng.normal(0, sigma), lo, hi))
        return ind

    # ---- vòng đời ----
    def run(self):
        tag = self.metric if self.metric != "tail" else f"tail_a{self.alpha:g}"
        if self.loss_weight:
            tag += "_lw"
        prefix = f"results/ga_{tag}"
        if self.seed != 42:
            prefix += f"_seed{self.seed}"
        log_f = open(f"{prefix}_log.jsonl", "w")
        t_start = time.time()

        pop = [self.random_individual() for _ in range(POP_SIZE)]
        fits = [self.fitness(ind) for ind in pop]
        best_fit, best_ind = min(fits), pop[int(np.argmin(fits))][:]
        stall = 0

        for gen in range(1, N_GENERATIONS + 1):
            order = np.argsort(fits)
            new_pop = [pop[i][:] for i in order[:ELITISM]]
            while len(new_pop) < POP_SIZE:
                p1, p2 = self.tournament(pop, fits), self.tournament(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < POP_SIZE:
                    new_pop.append(self.mutate(c2))
            pop = new_pop
            fits = [self.fitness(ind) for ind in pop]

            gen_best = min(fits)
            if gen_best < best_fit - 1e-6:
                best_fit, best_ind = gen_best, pop[int(np.argmin(fits))][:]
                stall = 0
            else:
                stall += 1

            rec = {"metric": self.metric, "gen": gen, "best_fitness": best_fit,
                   "gen_best": gen_best, "gen_mean": float(np.mean(fits)),
                   "best_cv_scores": self.scores_of(best_ind),
                   "n_evals": self.n_evals,
                   "elapsed_s": round(time.time() - t_start, 1),
                   "best_params": decode(best_ind, self.genes)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[{self.metric}] Gen {gen:2d} | best fitness {best_fit:.4f} "
                  f"| gen mean {rec['gen_mean']:.4f} | evals {self.n_evals} "
                  f"| {rec['elapsed_s']}s", flush=True)
            if stall >= PATIENCE:
                print(f"[{self.metric}] Dừng sớm: {PATIENCE} thế hệ không cải thiện.")
                break
        log_f.close()

        # Đánh giá cuối trên test
        best_params = decode(best_ind, self.genes)
        xgb_params, beta = split_params(best_params)
        m = XGBRegressor(tree_method="hist", random_state=42,
                         n_jobs=self.n_jobs, **xgb_params)
        t0 = time.time()
        sw = sample_weights(self.y_tr, beta) if beta is not None else None
        m.fit(self.X_tr, self.y_tr, sample_weight=sw)
        fit_s = time.time() - t0
        pred = m.predict(self.X_te)
        test = {
            "rmse": float(np.sqrt(mean_squared_error(self.y_te, pred))),
            "mae": float(mean_absolute_error(self.y_te, pred)),
            "r2": float(r2_score(self.y_te, pred)),
            "fit_seconds": round(fit_s, 2),
        }
        out = {"fitness_metric": self.metric, "best_params": best_params,
               "best_cv_scores": self.scores_of(best_ind),
               "test": test, "total_evals": self.n_evals,
               "total_seconds": round(time.time() - t_start, 1),
               "ga_config": {"pop": POP_SIZE, "max_gens": N_GENERATIONS,
                             "crossover": CROSSOVER_RATE, "mutation": MUTATION_RATE,
                             "elitism": ELITISM, "cv_folds": CV_FOLDS,
                             "seed": self.seed, "n_jobs": self.n_jobs,
                             "tail_alpha": self.alpha, "tail_bins": TAIL_BINS,
                             "loss_weight": self.loss_weight}}
        with open(f"{prefix}_best.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[{self.metric}] Best params:", best_params)
        print(f"[{self.metric}] Test: RMSE={test['rmse']:.4f} "
              f"MAE={test['mae']:.4f} R2={test['r2']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["rmse", "mae", "r2", "tail"], default="rmse")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="mức nhấn đuôi cho --metric tail (0=RMSE thường, 1=macro-RMSE)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--loss-weight", action="store_true",
                    help="thêm gene thứ 8: cường độ trọng số mẫu trong loss XGBoost")
    args = ap.parse_args()
    GA(args.metric, args.seed, args.n_jobs, alpha=args.alpha,
       loss_weight=args.loss_weight).run()
