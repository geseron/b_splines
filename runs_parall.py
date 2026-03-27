import itertools
import pandas as pd
import numpy as np
import torch
#from kan import KAN
from kan.utils import create_dataset_from_data, ex_round
from joblib import Parallel, delayed
import multiprocessing
import time
from logs import *
from save import *
import math
from tqdm import tqdm


n_knots_list = [4, 8, 12]
F_values = [0.3, 0.5, 0.8]
Cr_values = [0.4, 0.8]
n_repeats_per_comb = 4
pop_size = 20
generations = 50
noise_scale = 0.05


# === Новые настройки ===
n_repeats = 5  # Количество запусков на одну комбинацию
x_true = np.linspace(-2, 2, 500)



def run_single_experiment(args):
    (
        func, n_var, func_id, id_name,
        gap_range, n_samples, noise_level,
        width_tail, k, grid, seed
    ) = args
    

    row = {
        'func_id': id_name,
        'noise_level': noise_level,
        'gap_str': gap_str,
        'n_samples': n_samples,
        'width': str(width_current),
        'k': k,
    }

    return row



from joblib import Parallel, delayed
from tqdm import tqdm
import time
import os
import math

# Параметры
MAX_RETRIES = 3
TIMEOUT_SECONDS = 200
BATCH_SIZE = 36
n_jobs = 12

# Путь к папке с результатами
RESULTS_DIR = "results"

def is_batch_completed(id_name, batch_num):
    """
    Проверяет, выполнена ли пачка
    """
    parquet_path = f"{RESULTS_DIR}/results_{id_name}_{batch_num}.pkl"
    return os.path.isfile(parquet_path)


for func_idx, (func, n_var, name, id_name) in enumerate(FUNCTIONS):
    print(f"\n🚀 Запуск экспериментов для функции: {id_name}")
    v_log(f'Функция {id_name} начата')

    # Генерация всех задач (как раньше)
    tasks = []   # список всех задач для текущей функции
    for i in range(1,31):
        for n_knots in n_knots_list:
            for F in F_values:
                for Cr in Cr_values:
                    for _ in range(n_repeats_per_comb):
                        seed = np.random.randint(0, 2**30)
                        tasks.append((f'N{i}', f'datasets\\N{i}.pkl', n_knots, F, Cr, seed))

    total = len(tasks)
    print(f"Всего задач для {id_name}: {total}")
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Разбито на {n_batches} пачек по {BATCH_SIZE} задач")


    # Обработка пачками
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch_tasks = tasks[start_idx:end_idx]
        batch_num = batch_idx + 1
        batch_total = len(batch_tasks)

        # Проверяем, выполнена ли эта пачка
        if is_batch_completed(id_name, batch_num):
            print(f"✅ Пачка {batch_num}/{n_batches} уже выполнена (файл существует). Пропускаем.")
            v_log(f'{id_name} Пачка {batch_num} пропущена: файл уже есть')
            continue

        print(f"Обработка пачки {batch_num}/{n_batches} ({batch_total} задач)")


        # Цикл попыток с перезапуском при таймауте
        batch_results = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"!Попытка {attempt}/{MAX_RETRIES} для пачки {batch_num}")
                batch_results = Parallel(n_jobs=n_jobs, timeout=TIMEOUT_SECONDS, verbose=5)(
                    delayed(run_single_experiment)(task)
                    for task in tqdm(batch_tasks, desc=f"Пачка {batch_num} {id_name}", total=batch_total)
                )
                v_log(f'{id_name} Пачка {batch_num} завершена за {attempt} попытку. Результат: {len(batch_results)} записей')
                break  # Успех — выходим из цикла попыток

            except multiprocessing.TimeoutError:
            # Основной случай: таймаут joblib
                print(f"!⚠️ Таймаут! Пачка {batch_num}, попытка {attempt}")
                v_log(f'{id_name} Пачка {batch_num}, попытка {attempt} — TimeoutError')
                if attempt < MAX_RETRIES:
                    print(f"Перезапуск пачки {batch_num} (попытка {attempt + 1})...")
                else:
                    print(f"❌ Все {MAX_RETRIES} попыток исчерпаны для пачки {batch_num}. Пропускаем.")
                    v_log(f'{id_name} Пачка {batch_num}: все попытки неудачны, данных нет')
                    batch_results = []  # Пустой результат для дальнейшей обработки



        # Если результатов нет (все попытки провалились), пропускаем сохранение
        if batch_results is None or not batch_results:
            print(f"⚠️ Пачка {batch_num}: нет валидных результатов после {MAX_RETRIES} попыток")
            v_log(f'{id_name} Пачка {batch_num}: нет данных для сохранения')
            continue

        # Фильтрация None
        batch_rows = [r for r in batch_results if r is not None]
        v_log(f'{id_name} Пачка {batch_num} Фильтрация завершена. Строк: {len(batch_rows)}')


        if not batch_rows:
            print(f"⚠️ Пачка {batch_num}: нет валидных результатов после фильтрации")
            v_log(f'{id_name} Пачка {batch_num}: нет данных для сохранения после фильтрации')
            continue

        # Сохранение пачки
        batch_base_path = f"{RESULTS_DIR}/results_{id_name}_{batch_num}"
        save_result = save_experiment_data(
            rows=batch_rows,
            base_path=batch_base_path,
            #protocol=4,
            parquet_compression="snappy"
        )
        v_log(f'{id_name} Пачка {batch_num} сохранена: {save_result}')

        # Опционально: сохранение CSV
        # df_batch = pd.DataFrame(batch_rows)
        # csv_file = f"{RESULTS_DIR}/results_{id_name}_{batch_num}.csv"
        # df_batch.to_csv(csv_file, index=False, sep=';')
        # print(f"✅ Сохранено: {csv_file} ({len(batch_rows)} экспериментов)")
        # v_log(f'{id_name} Пачка {batch_num} CSV сохранён: {csv_file}')

print("Все функции обработаны!")
