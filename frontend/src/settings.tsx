import { useEffect, useState } from "react"
import { getSettings, saveSettings, suggestSettings } from "./api"
import { Modal } from "./modal"

function numberOf(raw: string): number | null {
  const value = Number(raw.trim().replace(",", "."))
  return Number.isFinite(value) ? value : null
}

function wholeOf(raw: string): number | null {
  const value = numberOf(raw)
  if (value === null || !Number.isInteger(value)) return null
  return value
}

export function SettingsDialog({ onClose }: { onClose: () => void }) {
  const [memory, setMemory] = useState("")
  const [asr, setAsr] = useState("")
  const [diar, setDiar] = useState("")
  const [threads, setThreads] = useState("")
  const [parallel, setParallel] = useState("")
  const [cores, setCores] = useState(1)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancel = false
    getSettings()
      .then((tune) => {
        if (cancel) return
        setAsr(String(tune.asrBatch))
        setDiar(String(tune.diarBatch))
        setThreads(String(tune.threads))
        setParallel(String(tune.stageParallel))
        setCores(tune.cores)
        setMemory(tune.memoryGb != null ? String(tune.memoryGb) : "6")
        setLoaded(true)
      })
      .catch((reason: unknown) => {
        if (!cancel) setError(reason instanceof Error ? reason.message : "Настройки не загрузились")
      })
    return () => {
      cancel = true
    }
  }, [])

  async function pick() {
    const gigabytes = numberOf(memory)
    if (gigabytes === null) {
      setError("Укажите, сколько гигабайт можно отдать")
      return
    }
    setBusy(true)
    setError(null)
    try {
      const suggestion = await suggestSettings(gigabytes)
      setMemory(String(gigabytes))
      setAsr(String(suggestion.asrBatch))
      setDiar(String(suggestion.diarBatch))
      setThreads(String(suggestion.threads))
      setParallel(String(suggestion.stageParallel))
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Не удалось подобрать")
    } finally {
      setBusy(false)
    }
  }

  async function save() {
    const asrBatch = wholeOf(asr)
    const diarBatch = wholeOf(diar)
    const threadCount = wholeOf(threads)
    const stageParallel = wholeOf(parallel)
    const memoryGb = memory.trim() === "" ? null : numberOf(memory)
    if (asrBatch === null || diarBatch === null || threadCount === null || stageParallel === null) {
      setError("Окна, потоки и число задач — целые числа")
      return
    }
    if (memory.trim() !== "" && memoryGb === null) {
      setError("Память: нужно число в гигабайтах")
      return
    }
    setBusy(true)
    setError(null)
    try {
      await saveSettings({ asrBatch, diarBatch, threads: threadCount, stageParallel, memoryGb })
      onClose()
    } catch (reason: unknown) {
      setError(reason instanceof Error ? reason.message : "Не удалось сохранить")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal title="Настройки разбора" onClose={onClose}>
      {!loaded && !error && <p className="hint">Загрузка…</p>}
      {loaded && (
      <div className="tune">
        <label>
          Сколько памяти отдать, ГБ
          <span className="tune-memory">
            <input
              className="modal-field"
              inputMode="decimal"
              value={memory}
              aria-label="Память, гигабайты"
              onChange={(event) => setMemory(event.target.value)}
            />
            <button type="button" className="ghost" disabled={busy} onClick={() => void pick()}>
              Подобрать
            </button>
          </span>
        </label>
        <label>
          <span className="tip">
            Окна распознавания
            <span className="usage-tip">Сколько фрагментов распознавание берёт за один проход. Больше — быстрее и тяжелее по памяти.</span>
          </span>
          <input
            className="modal-field"
            inputMode="numeric"
            value={asr}
            aria-label="Окна распознавания"
            onChange={(event) => setAsr(event.target.value)}
          />
        </label>
        <label>
          <span className="tip">
            Окна диаризации
            <span className="usage-tip">Сколько фрагментов диаризация берёт за один проход. Больше — быстрее и тяжелее по памяти.</span>
          </span>
          <input
            className="modal-field"
            inputMode="numeric"
            value={diar}
            aria-label="Окна диаризации"
            onChange={(event) => setDiar(event.target.value)}
          />
        </label>
        <label>
          <span className="tip">
            Потоков на всё, до {cores}
            <span className="usage-tip">Потолок ядер на все ролики.</span>
          </span>
          <input
            className="modal-field"
            inputMode="numeric"
            value={threads}
            aria-label="Потоки"
            onChange={(event) => setThreads(event.target.value)}
          />
        </label>
        <label>
          <span className="tip">
            Задач сразу, до {cores}
            <span className="usage-tip">Сколько роликов разбирать параллельно.</span>
          </span>
          <input
            className="modal-field"
            inputMode="numeric"
            value={parallel}
            aria-label="Задач сразу"
            onChange={(event) => setParallel(event.target.value)}
          />
        </label>
      </div>
      )}
      {error && <p className="bad">{error}</p>}
      <div className="modal-actions">
        <button type="button" className="ghost" onClick={onClose}>
          Отмена
        </button>
        <button type="button" className="send" disabled={!loaded || busy} onClick={() => void save()}>
          Сохранить
        </button>
      </div>
    </Modal>
  )
}
