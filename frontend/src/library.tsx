import { useEffect, useRef, useState } from "react"
import { Link } from "react-router-dom"
import { addByLink, holdVideo, listVideos, probeLink, removeVideo, renameVideo, uploadVideos, type StageChoice, type Video, type VideoUsage } from "./api"
import { bytes, clock, stageMark, STATUS, statusText, useNow } from "./format"
import { Modal } from "./modal"
import { SettingsDialog } from "./settings"
import { ThemeSwitch } from "./theme"

type Dialog =
  | { kind: "delete"; video: Video; error?: string }
  | { kind: "rename"; video: Video; draft: string; error?: string }

type LocalUpload = {
  key: string
  name: string
  progress: number
  status: "wait" | "send" | "error"
  begun: number
  error?: string
}

type Job = { key: string; file: File; stages: StageChoice }

type DraftFile = { key: string; file: File; duration: number | null; stages: StageChoice }

type LinkDraft = {
  url: string
  error?: string
  known: boolean
  looking: boolean
  duration: number | null
  stages: StageChoice
}

const SHORT = 90 * 60

function defaultStages(duration: number | null): StageChoice {
  const on = duration != null && duration > 0 && duration < SHORT
  return { diarize: on, merge: on, correct: on }
}

function readDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const node = document.createElement(file.type.startsWith("audio/") ? "audio" : "video")
    let settled = false
    const finish = (value: number | null) => {
      if (settled) return
      settled = true
      URL.revokeObjectURL(url)
      resolve(value)
    }
    node.preload = "metadata"
    node.onloadedmetadata = () => finish(Number.isFinite(node.duration) && node.duration > 0 ? node.duration : null)
    node.onerror = () => finish(null)
    window.setTimeout(() => finish(null), 4000)
    node.src = url
  })
}

const STAGE_FIELDS: { key: keyof StageChoice; label: string; title: string }[] = [
  { key: "diarize", label: "Диаризация", title: "Кто говорит" },
  { key: "merge", label: "Склейка фраз", title: "Собрать реплику из кусков" },
  { key: "correct", label: "Правка текста", title: "Поправить ошибки распознавания" },
]

function StageBoxes({ stages, onChange }: { stages: StageChoice; onChange: (next: StageChoice) => void }) {
  return (
    <div className="stage-opts">
      {STAGE_FIELDS.map((field) => (
        <label key={field.key} title={field.title}>
          <input
            type="checkbox"
            checked={stages[field.key]}
            onChange={(event) => onChange({ ...stages, [field.key]: event.target.checked })}
          />
          {field.label}
        </label>
      ))}
    </div>
  )
}

const USAGE_PARTS: { key: keyof Omit<VideoUsage, "total">; label: string }[] = [
  { key: "video", label: "Медиа" },
  { key: "files", label: "Файлы разбора" },
  { key: "text", label: "Текст" },
  { key: "vectors", label: "Векторы реплик" },
  { key: "indexes", label: "Индексы" },
]

function loadedSeconds(video: Video, now: number): number | null {
  const stages = video.stages ?? []
  if (stages.length) return stages.reduce((sum, stage) => sum + stage.seconds, 0)
  if (video.processedSec != null) return video.processedSec
  if (video.held || !video.startedAt || video.status === "queued") return null
  const start = Date.parse(video.startedAt)
  if (!Number.isFinite(start)) return null
  return Math.max(0, (now - start) / 1000)
}

function Loaded({ video, now }: { video: Video; now: number }) {
  const seconds = loadedSeconds(video, now)
  if (seconds == null) return null
  const stages = video.stages ?? []
  return (
    <span className="loaded">
      загружено за {clock(seconds)}
      {stages.length > 0 && (
        <span className="usage-tip" role="tooltip">
          {stages.map((stage) => (
            <span key={stage.stage}>
              {STATUS[stage.stage] ?? stage.stage}
              <b>{clock(stage.seconds)}</b>
            </span>
          ))}
        </span>
      )}
    </span>
  )
}

function Usage({ usage }: { usage: VideoUsage | undefined }) {
  if (!usage) return null
  const parts = USAGE_PARTS.filter((part) => usage[part.key] > 0)
  return (
    <span className="usage">
      {bytes(usage.total)}
      <span className="usage-tip" role="tooltip">
        {parts.length === 0 ? (
          <span>Пока пусто</span>
        ) : (
          parts.map((part) => (
            <span key={part.key}>
              {part.label}
              <b>{bytes(usage[part.key])}</b>
            </span>
          ))
        )}
      </span>
    </span>
  )
}

export function Library() {
  const [videos, setVideos] = useState<Video[]>([])
  const [locals, setLocals] = useState<LocalUpload[]>([])
  const [hot, setHot] = useState(false)
  const [note, setNote] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [dialog, setDialog] = useState<Dialog | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [link, setLink] = useState<LinkDraft | null>(null)
  const [drafts, setDrafts] = useState<DraftFile[] | null>(null)
  const [linkBusy, setLinkBusy] = useState(false)
  const jobs = useRef<Job[]>([])
  const pumping = useRef(false)
  const ticking = locals.length > 0 || videos.some((video) => video.status !== "ready" && video.status !== "error")
  const now = useNow(ticking)

  useEffect(() => {
    let cancel = false
    const tick = () => {
      listVideos()
        .then((data) => {
          if (cancel) return
          setVideos(data.videos)
          setLoadError(null)
        })
        .catch((error: unknown) => {
          if (!cancel) setLoadError(error instanceof Error ? error.message : "Список не загрузился")
        })
    }
    tick()
    const timer = window.setInterval(tick, 2000)
    return () => {
      cancel = true
      window.clearInterval(timer)
    }
  }, [])

  async function refresh() {
    const data = await listVideos()
    setVideos(data.videos)
  }

  async function pump() {
    if (pumping.current) return
    pumping.current = true
    try {
      while (jobs.current.length) {
        const job = jobs.current.shift()
        if (!job) break
        setLocals((current) =>
          current.map((item) =>
            item.key === job.key ? { ...item, status: "send", progress: 0, begun: Date.now() } : item,
          ),
        )
        try {
          await uploadVideos([job.file], job.stages, (ratio) => {
            setLocals((current) =>
              current.map((item) => (item.key === job.key ? { ...item, progress: ratio } : item)),
            )
          })
          setLocals((current) => current.filter((item) => item.key !== job.key))
          await refresh()
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : "Не удалось загрузить"
          setLocals((current) =>
            current.map((item) => (item.key === job.key ? { ...item, status: "error", error: message } : item)),
          )
        }
      }
    } finally {
      pumping.current = false
      if (jobs.current.length) void pump()
    }
  }

  async function prepare(files: File[]) {
    if (!files.length) return
    setNote(null)
    const next = await Promise.all(
      files.map(async (file) => {
        const duration = await readDuration(file)
        return { key: crypto.randomUUID(), file, duration, stages: defaultStages(duration) }
      }),
    )
    setDrafts((current) => [...(current ?? []), ...next])
  }

  function confirmDrafts() {
    if (!drafts?.length) return
    const next = drafts.map((item) => ({ key: item.key, file: item.file, stages: item.stages }))
    jobs.current.push(...next)
    setLocals((current) => [
      ...current,
      ...next.map((job) => ({
        key: job.key,
        name: job.file.name,
        progress: 0,
        status: "wait" as const,
        begun: Date.now(),
      })),
    ])
    setDrafts(null)
    void pump()
  }

  async function lookUpLink() {
    if (!link || linkBusy) return
    const url = link.url.trim()
    if (!url) {
      setLink({ ...link, error: "Нужна ссылка" })
      return
    }
    setLinkBusy(true)
    setLink({ ...link, url, error: undefined, looking: true })
    try {
      const found = await probeLink(url)
      setLink({
        url,
        known: true,
        looking: false,
        duration: found.durationSec,
        stages: defaultStages(found.durationSec),
      })
    } catch (error: unknown) {
      setLink({
        url,
        known: true,
        looking: false,
        duration: null,
        stages: defaultStages(null),
        error: error instanceof Error ? error.message : "Длина не прочиталась",
      })
    } finally {
      setLinkBusy(false)
    }
  }

  async function confirmLink() {
    if (!link || linkBusy) return
    if (!link.known) {
      await lookUpLink()
      return
    }
    const url = link.url.trim()
    setLinkBusy(true)
    try {
      await addByLink(url, link.stages)
      await refresh()
      setLink(null)
    } catch (error: unknown) {
      setLink({ ...link, error: error instanceof Error ? error.message : "Ссылка не принята" })
    } finally {
      setLinkBusy(false)
    }
  }

  async function toggleHold(video: Video) {
    setNote(null)
    try {
      const updated = await holdVideo(video.id, !video.held)
      setVideos((current) => current.map((item) => (item.id === updated.id ? updated : item)))
    } catch (error: unknown) {
      setNote(error instanceof Error ? error.message : "Не удалось поставить на паузу")
    }
  }

  async function confirmDelete() {
    if (!dialog || dialog.kind !== "delete") return
    try {
      await removeVideo(dialog.video.id)
      setVideos((current) => current.filter((item) => item.id !== dialog.video.id))
      setDialog(null)
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "Не удалось удалить ролик"
      setDialog({ ...dialog, error: message })
    }
  }

  async function confirmRename() {
    if (!dialog || dialog.kind !== "rename") return
    const title = dialog.draft.trim()
    if (!title) {
      setDialog({ ...dialog, error: "Нужно название" })
      return
    }
    if (title === dialog.video.title) {
      setDialog(null)
      return
    }
    try {
      const updated = await renameVideo(dialog.video.id, title)
      setVideos((current) => current.map((item) => (item.id === updated.id ? { ...item, title: updated.title } : item)))
      setDialog(null)
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : "Не удалось переименовать"
      setDialog({ ...dialog, error: message })
    }
  }

  const active = locals.find((item) => item.status === "send")

  return (
    <main
      className="lib"
      onDragOver={(event) => {
        event.preventDefault()
        setHot(true)
      }}
      onDragLeave={() => setHot(false)}
      onDrop={(event) => {
        event.preventDefault()
        setHot(false)
        void prepare(Array.from(event.dataTransfer.files))
      }}
    >
      <div className="lib-top">
        <p className="mark">Разбор</p>
        <div className="lib-actions">
          <button type="button" className="tool-btn" onClick={() => setSettingsOpen(true)}>
            Настройки
          </button>
          <ThemeSwitch />
        </div>
      </div>
      <h1>Медиа</h1>

      <div className={hot ? "uploader hot" : "uploader"}>
        <label className="pick">
          Вручную
          <input
            type="file"
            accept="video/*,audio/*,.mkv,.webm,.mov,.avi,.m4v,.mp3,.wav,.m4a,.aac,.ogg,.flac,.opus,.wma"
            multiple
            hidden
            onChange={(event) => {
              const chosen = event.target.files ? Array.from(event.target.files) : []
              event.target.value = ""
              void prepare(chosen)
            }}
          />
        </label>
        <button type="button" className="pick" onClick={() => setLink({ url: "", known: false, looking: false, duration: null, stages: defaultStages(null) })}>
          По ссылке
        </button>
        <p>
          {active
            ? `Загрузка «${active.name}» ${Math.round(active.progress * 100)}% · ${clock(Math.max(0, (now - active.begun) / 1000))}`
            : "Или перетащите файлы сюда."}
        </p>
      </div>
      {active && (
        <div className="bar" aria-hidden="true">
          <div style={{ width: `${Math.round(active.progress * 100)}%` }} />
        </div>
      )}
      {locals.length > 0 && (
        <ul className="uploads">
          {locals.map((item) => (
            <li key={item.key}>
              <span className="clamp">{item.name}</span>
              <span className={item.status === "error" ? "bad" : "live"}>
                {item.status === "send"
                  ? `${Math.round(item.progress * 100)}% · ${clock(Math.max(0, (now - item.begun) / 1000))}`
                  : item.status === "error"
                    ? item.error || "ошибка"
                    : `ждёт · ${clock(Math.max(0, (now - item.begun) / 1000))}`}
              </span>
            </li>
          ))}
        </ul>
      )}
      {note && <p className="bad">{note}</p>}
      {loadError && <p className="bad">{loadError}</p>}

      <ul className="videos">
        {videos.map((video) => {
          const busy = video.status !== "ready" && video.status !== "error"
          return (
            <li key={video.id}>
              <Link to={`/v/${video.id}`}>
                <span>
                  <span className="clip">
                    <strong>{video.title}</strong>
                    <span className="usage-tip">{video.title}</span>
                  </span>
                  <span className="file-meta">
                    <Usage usage={video.usage} />
                    <Loaded video={video} now={now} />
                  </span>
                  {video.status !== "ready" && video.status !== "error" && (
                    <span className={video.held ? "meter" : "meter live"} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={video.progress}>
                      <span style={{ width: `${video.progress}%` }} />
                    </span>
                  )}
                  {video.error && <small className="bad clamp">{video.error}</small>}
                </span>
                <span className={video.status === "error" ? "bad" : video.status === "ready" || video.held ? "muted" : "live"}>
                  {video.durationSec ? `${clock(video.durationSec)} · ` : ""}
                  {statusText(video)}
                  {stageMark(video)}
                </span>
              </Link>
              <div className="row-actions">
                {busy && (
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => void toggleHold(video)}
                  >
                    {video.held ? "Продолжить" : "Пауза"}
                  </button>
                )}
                <button
                  type="button"
                  className="ghost"
                  onClick={() => setDialog({ kind: "rename", video, draft: video.title })}
                >
                  Переименовать
                </button>
                <button type="button" className="ghost" onClick={() => setDialog({ kind: "delete", video })}>
                  Удалить
                </button>
              </div>
            </li>
          )
        })}
      </ul>
      {videos.length === 0 && locals.length === 0 && !loadError && <p className="muted empty">Пока пусто.</p>}
      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
      {link && (
        <Modal title="По ссылке" onClose={() => { if (!linkBusy) setLink(null) }}>
          <input
            className="modal-field"
            value={link.url}
            autoFocus
            placeholder="https://"
            aria-label="Ссылка на медиа"
            onChange={(event) => setLink({ url: event.target.value, known: false, looking: false, duration: null, stages: defaultStages(null) })}
            onKeyDown={(event) => {
              if (event.key === "Enter") void confirmLink()
            }}
          />
          {link.known && (
            <>
              <p className="muted">{link.duration != null ? clock(link.duration) : "Длина неизвестна"}</p>
              <StageBoxes stages={link.stages} onChange={(stages) => setLink({ ...link, stages })} />
            </>
          )}
          {link.error && <p className="bad">{link.error}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" disabled={linkBusy} onClick={() => setLink(null)}>
              Отмена
            </button>
            <button type="button" className="send" disabled={linkBusy} onClick={() => void confirmLink()}>
              {link.known ? "Скачать" : "Дальше"}
            </button>
          </div>
        </Modal>
      )}
      {drafts && (
        <Modal title="Добавить" wide onClose={() => setDrafts(null)}>
          <ul className="drafts">
            {drafts.map((item) => (
              <li key={item.key}>
                <span className="clamp">{item.file.name}</span>
                <span className="muted">{item.duration != null ? clock(item.duration) : "длина неизвестна"}</span>
                <StageBoxes
                  stages={item.stages}
                  onChange={(stages) =>
                    setDrafts((current) =>
                      current?.map((row) => (row.key === item.key ? { ...row, stages } : row)) ?? null,
                    )
                  }
                />
              </li>
            ))}
          </ul>
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => setDrafts(null)}>
              Отмена
            </button>
            <button type="button" className="send" onClick={confirmDrafts}>
              Добавить
            </button>
          </div>
        </Modal>
      )}
      {dialog?.kind === "delete" && (
        <Modal title="Удалить ролик?" onClose={() => setDialog(null)}>
          <p>Вместе с «{dialog.video.title}» пропадут разбор и поиск.</p>
          {dialog.error && <p className="bad">{dialog.error}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => setDialog(null)}>
              Отмена
            </button>
            <button type="button" className="danger" onClick={() => void confirmDelete()}>
              Удалить
            </button>
          </div>
        </Modal>
      )}
      {dialog?.kind === "rename" && (
        <Modal title="Переименовать" wide onClose={() => setDialog(null)}>
          <input
            className="modal-field"
            value={dialog.draft}
            autoFocus
            aria-label="Новое название"
            onChange={(event) => setDialog({ ...dialog, draft: event.target.value, error: undefined })}
            onKeyDown={(event) => {
              if (event.key === "Enter") void confirmRename()
            }}
          />
          {dialog.error && <p className="bad">{dialog.error}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => setDialog(null)}>
              Отмена
            </button>
            <button type="button" className="send" onClick={() => void confirmRename()}>
              Сохранить
            </button>
          </div>
        </Modal>
      )}
    </main>
  )
}
