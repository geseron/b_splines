import numpy as np
import pickle
import os
import math
import itertools
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ClassDivEvo import DivEvo
from bspline import BSplineRegression

# ========== 1. Параметры эксперимента ==========
degree = 3
reg = 1e-6

n_knots_list = [4, 8, 12]
F_values = [0.3, 0.5, 0.8]
Cr_values = [0.4, 0.8]
n_repeats_per_comb = 10
pop_size = 20
generations = 500
noise_scale = 0.05

DATASET_DIR = "datasets"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========== 2. Целевая функция ==========
def objective(knots, x_train, y_train, x_val, y_val, degree=3, eps=0.01):
    original = knots.copy()
    knots_sorted = np.sort(knots)
    order_penalty = np.sum((original - knots_sorted)**2)
    boundary_penalty = np.sum(np.maximum(0, original - 1)**2) + np.sum(np.maximum(0, -original)**2)

    # слияние слишком близких узлов
    merged = []
    i = 0
    while i < len(knots_sorted):
        j = i
        while j + 1 < len(knots_sorted) and knots_sorted[j+1] - knots_sorted[j] < eps:
            j += 1
        if j > i:
            merged.append(np.mean(knots_sorted[i:j+1]))
        else:
            merged.append(knots_sorted[i])
        i = j + 1
    knots_sorted = np.sort(merged)
    penalty = 100.0 * (order_penalty + boundary_penalty)
    try:
        model = BSplineRegression(degree=degree, knots=knots_sorted, regularization=1e-6)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        return mse + penalty
    except:
        return 1e10

# ========== 3. Эксперимент для одного набора параметров ==========
def run_experiment(func_id, data_path, n_knots, F, Cr, seed):
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        x = data["x"]
        y_noisy = data["y_noisy"]
    except Exception as e:
        print(f"Ошибка загрузки {data_path}: {e}")
        return None

    x_train, x_val, y_train, y_val = train_test_split(x, y_noisy, test_size=0.3, random_state=seed)

    def obj_wrapper(knots):
        return objective(knots, x_train, y_train, x_val, y_val, degree)

    uniform_knots = np.linspace(0, 1, n_knots + 2)[1:-1]
    start_pop = []
    for _ in range(pop_size):
        knots = uniform_knots + np.random.normal(0, noise_scale, size=n_knots)
        knots = np.clip(knots, 0, 1)
        start_pop.append(np.sort(knots))

    evo = DivEvo(
        f=F,
        cr=Cr,
        g=generations,
        fofma=obj_wrapper,
        start_population=start_pop,
        mutation='rand_to_best',
        lowlimit=0,
        uplimit=1
    )
    try:
        evo.findOptim()
    except Exception as e:
        print(f"Ошибка DivEvo для {func_id}, n_knots={n_knots}, F={F}, Cr={Cr}, seed={seed}: {e}")
        return None

    best_knots = evo.result["Point"]
    best_mse = evo.result["Minimum"]
    best_knots = np.sort(best_knots)
    if np.any(best_knots < 0) or np.any(best_knots > 1):
        return None

    return {
        "func_id": func_id,
        "n_knots": n_knots,
        "F": F,
        "Cr": Cr,
        "seed": seed,
        "knots": best_knots.tolist(),
        "mse": best_mse
    }

# ========== 4. Генерация задач для каждой функции ==========
func_ids = [f"N{i}" for i in range(1, 31)]
data_files = {fid: os.path.join(DATASET_DIR, f"{fid}.pkl") for fid in func_ids}
missing = [fid for fid, path in data_files.items() if not os.path.exists(path)]
if missing:
    print(f"Отсутствуют файлы данных: {missing}")
    exit(1)

# ========== 5. Обработка функций ==========
MAX_RETRIES = 3
TIMEOUT_SECONDS = 500
BATCH_SIZE = 46
N_JOBS = 12

# Считаем общее количество пачек для прогресса
total_batches = 0
for fid in func_ids:
    total_tasks = len(n_knots_list) * len(F_values) * len(Cr_values) * n_repeats_per_comb
    total_batches += math.ceil(total_tasks / BATCH_SIZE)
print(f"Всего пачек: {total_batches}")

processed = 0
for fid in func_ids:
    data_path = data_files[fid]
    # Генерация всех задач для этой функции
    tasks = []
    for n_knots in n_knots_list:
        for F in F_values:
            for Cr in Cr_values:
                for _ in range(n_repeats_per_comb):
                    seed = np.random.randint(0, 2**30)
                    tasks.append((fid, data_path, n_knots, F, Cr, seed))

    n_batches = math.ceil(len(tasks) / BATCH_SIZE)

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(tasks))
        batch_tasks = tasks[start:end]
        batch_num = batch_idx + 1

        batch_filename = f"{fid}_{batch_num}.pkl"
        batch_path = os.path.join(RESULTS_DIR, batch_filename)

        if os.path.exists(batch_path):
            processed += 1
            print(f"Прогресс: {processed}/{total_batches} – уже есть {batch_path}")
            continue

        print(f"Обработка {fid}, пачка {batch_num}/{n_batches}")

        # Повторные попытки
        batch_results = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                batch_results = Parallel(n_jobs=N_JOBS, timeout=TIMEOUT_SECONDS)(
                    delayed(run_experiment)(*task) for task in batch_tasks
                )
                print(f"Пачка {batch_num} для {fid} завершена")
                break
            except multiprocessing.TimeoutError:
                print(f"Таймаут пачки {batch_num} для {fid}, попытка {attempt}")
                if attempt == MAX_RETRIES:
                    batch_results = []

        if not batch_results:
            print(f"Нет результатов для {batch_filename}")
            processed += 1
            continue

        valid_rows = [r for r in batch_results if r is not None]
        if not valid_rows:
            print(f"Нет валидных результатов для {batch_filename}")
            processed += 1
            continue

        with open(batch_path, "wb") as f:
            pickle.dump(valid_rows, f)
        print(f"Сохранён {batch_path} ({len(valid_rows)} записей)")
        processed += 1

print("Все эксперименты завершены!")