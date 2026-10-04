"""One inference job. Window counts come from the settings modal."""

from __future__ import annotations

import os

CORES = max(1, os.cpu_count() or 1)
DEFAULT_MEMORY_GB = 6.0
ASR_LIMIT = 32
DIAR_LIMIT = 16
# Автоподбор не поднимает диаризацию выше этого: 16 окон уже роняли контейнер.
DIAR_AUTO_LIMIT = 8
MEMORY_LIMIT = 128


class Runtime:
    asr_batch: int = 4
    diar_batch: int = 1
    threads: int = min(CORES, 4)
    stage_parallel: int = 1
    memory_gb: float | None = DEFAULT_MEMORY_GB


runtime = Runtime()


def seen_memory_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def _whole(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: нужно целое число")
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"{label}: нужно целое число")
    whole = int(number)
    if whole < low or whole > high:
        raise ValueError(f"{label}: от {low} до {high}")
    return whole


def normalize(
    asr_batch: object,
    diar_batch: object,
    threads: object,
    memory_gb: object,
    stage_parallel: object,
) -> dict:
    asr = _whole(asr_batch, "Окна распознавания", 1, ASR_LIMIT)
    diar = _whole(diar_batch, "Окна диаризации", 1, DIAR_LIMIT)
    threads_n = _whole(threads, "Потоки", 1, CORES)
    parallel = _whole(stage_parallel, "Задач сразу", 1, CORES)
    memory: float | None
    if memory_gb is None or memory_gb == "":
        memory = None
    else:
        if isinstance(memory_gb, bool) or not isinstance(memory_gb, (int, float)):
            raise ValueError("Память: нужно число в гигабайтах")
        memory = float(memory_gb)
        if memory < 1 or memory > MEMORY_LIMIT:
            raise ValueError(f"Память: от 1 до {MEMORY_LIMIT} ГБ")
        memory = round(memory, 1)
    return {
        "asrBatch": asr,
        "diarBatch": diar,
        "threads": threads_n,
        "stageParallel": parallel,
        "memoryGb": memory,
    }


def suggest(memory_gb: object) -> dict:
    """Окна от памяти, которую можно отдать контейнерам.

    3 ГБ остаются базе, хранилищу и системе. На оставшихся ~11 ГБ
    (всего 14) выходит 2 окна диаризации и 8 окон распознавания —
    те числа, на которых один ролик уже помещался.
    """
    if isinstance(memory_gb, bool) or not isinstance(memory_gb, (int, float)):
        raise ValueError("Память: нужно число в гигабайтах")
    memory = float(memory_gb)
    if memory < 1 or memory > MEMORY_LIMIT:
        raise ValueError(f"Память: от 1 до {MEMORY_LIMIT} ГБ")
    usable = max(0.0, memory - 3.0)
    diar = int(round((usable - 4.0) / 3.5))
    diar = max(1, min(DIAR_AUTO_LIMIT, diar))
    asr = max(2, min(16, diar * 4))
    if usable < 6:
        threads = min(CORES, 4)
        parallel = min(2, CORES)
    elif usable < 10:
        threads = min(CORES, 8)
        parallel = min(4, CORES)
    else:
        threads = CORES
        parallel = min(8, CORES)
    return {"asrBatch": asr, "diarBatch": diar, "threads": threads, "stageParallel": parallel}


def starting_tune() -> dict:
    picked = suggest(DEFAULT_MEMORY_GB)
    return {**picked, "memoryGb": DEFAULT_MEMORY_GB}


def apply(values: dict) -> None:
    runtime.asr_batch = int(values["asrBatch"])
    runtime.diar_batch = int(values["diarBatch"])
    runtime.threads = int(values["threads"])
    runtime.stage_parallel = int(values.get("stageParallel") or 1)
    runtime.memory_gb = values.get("memoryGb")
    count = str(runtime.threads)
    os.environ["OMP_NUM_THREADS"] = count
    os.environ["MKL_NUM_THREADS"] = count
    os.environ["OPENBLAS_NUM_THREADS"] = count
    os.environ["NUMEXPR_NUM_THREADS"] = count
    print(
        f"процессор: {runtime.threads} потоков, распознавание {runtime.asr_batch}, диаризация {runtime.diar_batch}, параллельность {runtime.stage_parallel}",
        flush=True,
    )


def bind_threads(count: int) -> None:
    """Used by the worker process. The API never loads torch."""
    count = max(1, int(count))
    os.environ["OMP_NUM_THREADS"] = str(count)
    os.environ["MKL_NUM_THREADS"] = str(count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(count)
    os.environ["NUMEXPR_NUM_THREADS"] = str(count)
    import torch

    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = True


def configure() -> None:
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["OMP_WAIT_POLICY"] = "ACTIVE"
    apply(starting_tune())


CHUNK_STEPS = (300, 150, 90, 60)


def chunk_seconds(pressure: int) -> int:
    index = min(max(0, int(pressure)), len(CHUNK_STEPS) - 1)
    return CHUNK_STEPS[index]


def lighter(level: int, asr: int, diar: int, threads: int) -> dict:
    """Per-task degradation. Global settings stay untouched."""
    step = min(max(0, int(level)), len(CHUNK_STEPS) - 1)
    divisor = 2 ** min(max(0, int(level)), 6)
    return {
        "chunk": CHUNK_STEPS[step],
        "asr": max(1, int(asr) // divisor),
        "diar": max(1, int(diar) // divisor),
        "threads": max(1, int(threads) // divisor),
        "embedBatch": max(1, 16 // divisor),
    }


def eased(values: dict) -> dict:
    """One step down after a crash. Floors at a single window and one thread."""

    def cut(name: str) -> int:
        current = int(values[name])
        if current <= 1:
            return 1
        return max(1, current // 2)

    return {
        "asrBatch": cut("asrBatch"),
        "diarBatch": cut("diarBatch"),
        "threads": cut("threads"),
        "stageParallel": cut("stageParallel"),
        "memoryGb": values.get("memoryGb"),
    }


def available_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def snapshot() -> dict:
    return {
        "asrBatch": runtime.asr_batch,
        "diarBatch": runtime.diar_batch,
        "threads": runtime.threads,
        "stageParallel": runtime.stage_parallel,
        "memoryGb": runtime.memory_gb,
        "seenMemoryGb": seen_memory_gb(),
        "cores": CORES,
    }
