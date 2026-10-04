import { useEffect, useRef, useState, type ReactNode } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import {
  askAgent,
  askVideo,
  stopDialog,
  buildMoments,
  clearAgent,
  clearChat,
  createIndex,
  getDialog,
  getVideo,
  holdVideo,
  listIndexes,
  removeIndex,
  removeVideo,
  renameSpeaker,
  type AgentMessage,
  type ChatMessage,
  type Detail,
  type SearchIndex,
  type Segment,
  type Speaker,
} from "./api"
import { clock, originalUrl, runningClock, speakerHue, stageMark, statusText, useNow } from "./format"
import { labelsToMentions, mentionAt, mentionPattern } from "./mentions"
import { Modal } from "./modal"
import { Panes } from "./panes"

const AGENT_VERSIONS = [
  { id: "bear", label: "agent-bear" },
  { id: "mouse", label: "agent-mouse" },
  { id: "zebra", label: "agent-zebra" },
]

function liveActionStatus(actions: { title: string; detail: string; progress?: number }[]): string {
  const last = actions[actions.length - 1]
  if (!last) return ""
  if (last.title === "Временный индекс" || typeof last.progress === "number") {
    const marked = last.detail.match(/^«([^»]+)»/)
    const name = (marked?.[1] || last.detail).trim()
    return name ? `Строим «${name.slice(0, 48)}»...` : "Строим индекс..."
  }
  if (last.title === "Мысль") return last.detail.trim()
  const named: Record<string, string> = {
    "Поиск по репликам": "Ищу по репликам...",
    "Обход текста": "Ищу по словам...",
    "Чтение отрезка": "Читаю отрезок...",
    "Поиск по индексу": "Ищу по индексу...",
    "Чтение индекса": "Читаю индекс...",
    Спикеры: "Смотрю спикеров...",
    Думает: "Думает...",
  }
  return named[last.title] || last.detail.trim() || last.title
}

function storedAgentVersion(): string {
  const saved = localStorage.getItem("va-agent-version")
  if (saved === "2" || saved === "analytic") return "bear"
  if (saved && AGENT_VERSIONS.some((version) => version.id === saved)) return saved
  return AGENT_VERSIONS[AGENT_VERSIONS.length - 1].id
}

function readVolume(): number {
  const raw = localStorage.getItem("va-volume")
  if (!raw) return 1
  const saved = Number(raw)
  if (!Number.isFinite(saved)) return 1
  return Math.min(1, Math.max(0, saved))
}

type AnswerVoice = "f" | "m"

function storedVoice(): AnswerVoice {
  return localStorage.getItem("va-voice") === "f" ? "f" : "m"
}

function storedSpeechVolume(): number {
  const raw = localStorage.getItem("va-voice-volume")
  if (!raw) return 1
  const value = Number(raw)
  if (!Number.isFinite(value)) return 1
  return Math.min(1, Math.max(0, value))
}

type SpeechPhase = { token: string; phase: "wait" | "play" }

let speechGen = 0
let speechAbort: AbortController | null = null
let speechCtx: AudioContext | null = null
let speechGain: GainNode | null = null
let speechLevel = storedSpeechVolume()
let speechSource: AudioBufferSourceNode | null = null
let speechPlayer: HTMLAudioElement | null = null
let speechLink: string | null = null
let speechNow: SpeechPhase | null = null
const speechListeners = new Set<() => void>()

function unlockSpeech() {
  const Ctx = window.AudioContext
  if (!Ctx) return
  if (!speechCtx) speechCtx = new Ctx()
  if (speechCtx.state !== "closed") void speechCtx.resume()
}

function applySpeechVolume(level: number) {
  speechLevel = Math.min(1, Math.max(0, level))
  if (speechGain) speechGain.gain.value = speechLevel
  if (speechPlayer) speechPlayer.volume = speechLevel
}

function speechOutput(): AudioNode {
  if (!speechCtx) throw new Error("нет звука")
  if (!speechGain) {
    speechGain = speechCtx.createGain()
    speechGain.connect(speechCtx.destination)
  }
  speechGain.gain.value = speechLevel
  return speechGain
}

function emitSpeech() {
  speechListeners.forEach((listener) => listener())
}

function haltSpeech() {
  speechAbort?.abort()
  speechAbort = null
  if (speechSource) {
    speechSource.onended = null
    try {
      speechSource.stop()
    } catch {
      /* источник уже остановлен */
    }
    speechSource = null
  }
  if (speechPlayer) {
    speechPlayer.onended = null
    speechPlayer.pause()
    speechPlayer = null
  }
  if (speechLink) {
    URL.revokeObjectURL(speechLink)
    speechLink = null
  }
}

function stopSpeech() {
  speechGen += 1
  haltSpeech()
  speechNow = null
  emitSpeech()
}

async function startSpeech(text: string, token: string, voice: AnswerVoice) {
  stopSpeech()
  const generation = speechGen
  const controller = new AbortController()
  speechAbort = controller
  speechNow = { token, phase: "wait" }
  emitSpeech()
  try {
    const response = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voice }),
      signal: controller.signal,
    })
    if (!response.ok) throw new Error(String(response.status))
    const blob = await response.blob()
    if (generation !== speechGen) return
    if (speechCtx && speechCtx.state === "running") {
      const decoded = await speechCtx.decodeAudioData(await blob.arrayBuffer())
      if (generation !== speechGen) return
      const source = speechCtx.createBufferSource()
      source.buffer = decoded
      source.connect(speechOutput())
      speechSource = source
      source.onended = () => {
        if (generation !== speechGen) return
        speechSource = null
        speechNow = null
        emitSpeech()
      }
      source.start()
      speechNow = { token, phase: "play" }
      emitSpeech()
      return
    }
    const url = URL.createObjectURL(blob)
    speechLink = url
    const player = new Audio(url)
    player.volume = speechLevel
    speechPlayer = player
    player.onended = () => {
      if (generation !== speechGen) return
      haltSpeech()
      speechNow = null
      emitSpeech()
    }
    await player.play()
    if (generation !== speechGen) {
      haltSpeech()
      return
    }
    speechNow = { token, phase: "play" }
    emitSpeech()
  } catch {
    if (generation !== speechGen) return
    haltSpeech()
    speechNow = null
    emitSpeech()
  }
}

function SpeakButton({ text, token, voice }: { text: string; token: string; voice: AnswerVoice }) {
  const [, redraw] = useState(0)
  useEffect(() => {
    const redrawNow = () => redraw((value) => value + 1)
    speechListeners.add(redrawNow)
    return () => {
      speechListeners.delete(redrawNow)
    }
  }, [])
  const phase = speechNow?.token === token ? speechNow.phase : "idle"
  const title = phase === "play" ? "Остановить" : "Слушать"
  return (
    <button
      type="button"
      className={phase === "idle" ? "speak" : `speak ${phase === "play" ? "on" : "wait"}`}
      title={title}
      aria-label={title}
      onClick={(event) => {
        event.stopPropagation()
        if (phase === "idle") {
          unlockSpeech()
          void startSpeech(text, token, voice)
        } else stopSpeech()
      }}
    >
      {phase === "play" ? (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="4" y="4" width="8" height="8" rx="1.2" />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3 6.3h1.8L8.2 3.6v8.8L4.8 9.7H3z" />
          <path d="M10.4 6.2a2.4 2.4 0 0 1 0 3.6" />
          <path d="M12 4.7a4.6 4.6 0 0 1 0 6.6" />
        </svg>
      )}
    </button>
  )
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      const area = document.createElement("textarea")
      area.value = text
      area.setAttribute("readonly", "")
      area.style.position = "fixed"
      area.style.left = "-9999px"
      document.body.append(area)
      area.select()
      document.execCommand("copy")
      area.remove()
    }
    setDone(true)
    window.setTimeout(() => setDone(false), 1400)
  }
  return (
    <button
      type="button"
      className="copy"
      title={done ? "Скопировано" : "Скопировать"}
      aria-label="Скопировать"
      onClick={(event) => {
        event.stopPropagation()
        void copy()
      }}
    >
      {done ? (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <path d="M3 8.2 6.2 11.4 13 4.6" />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16" aria-hidden="true">
          <rect x="5.5" y="2.5" width="8" height="10" rx="1.2" />
          <path d="M3.5 5.5h-1a1 1 0 0 0-1 1v7a1 1 0 0 0 1 1h7a1 1 0 0 0 1-1v-1" />
        </svg>
      )}
    </button>
  )
}

const TIME_MARK = /@(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})-(\d{1,2}:\d{2}:\d{2}|\d{1,3}:\d{2})/g

function markSeconds(mark: string): number {
  const parts = mark.split(":").map(Number)
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]
  return parts[0] * 60 + parts[1]
}

function MentionBits({ text, speakers }: { text: string; speakers: Speaker[] }) {
  const shown = labelsToMentions(text, speakers)
  const pattern = mentionPattern(speakers)
  if (!pattern) return <>{shown}</>
  const parts: ReactNode[] = []
  let cursor = 0
  for (const match of shown.matchAll(pattern)) {
    const index = match.index ?? 0
    if (index < cursor) continue
    if (index > cursor) parts.push(shown.slice(cursor, index))
    parts.push(
      <span className="mention" key={`${index}-${match[0]}`}>
        {match[0]}
      </span>,
    )
    cursor = index + match[0].length
  }
  if (cursor < shown.length) parts.push(shown.slice(cursor))
  return <>{parts}</>
}

function MessageText({
  text,
  speakers,
  onSeek,
}: {
  text: string
  speakers: Speaker[]
  onSeek?: (seconds: number) => void
}) {
  const shown = labelsToMentions(text, speakers)
  const pattern = mentionPattern(speakers)
  const parts: ReactNode[] = []
  let cursor = 0
  const tokens = [...shown.matchAll(TIME_MARK)]
  if (pattern) {
    for (const match of shown.matchAll(pattern)) tokens.push(match)
  }
  tokens.sort((left, right) => (left.index ?? 0) - (right.index ?? 0))
  for (const match of tokens) {
    const index = match.index ?? 0
    if (index < cursor) continue
    if (index > cursor) parts.push(shown.slice(cursor, index))
    if (match[0].includes("-")) {
      const start = markSeconds(match[1])
      parts.push(
        <button
          key={`${index}-${match[0]}`}
          type="button"
          className="jump"
          onClick={() => onSeek?.(start)}
        >
          {match[0]}
        </button>,
      )
    } else {
      parts.push(
        <span className="mention" key={`${index}-${match[0]}`}>
          {match[0]}
        </span>,
      )
    }
    cursor = index + match[0].length
  }
  if (cursor < shown.length) parts.push(shown.slice(cursor))
  return <p>{parts}</p>
}

type IndexPicks = { lines: boolean; known: Record<string, boolean> }

function readPicks(videoId: string): IndexPicks | null {
  const raw = localStorage.getItem(`va-indexes:${videoId}`)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as IndexPicks
    if (typeof parsed.lines !== "boolean" || !parsed.known || typeof parsed.known !== "object") return null
    return parsed
  } catch {
    return null
  }
}

function chosenIds(catalog: SearchIndex[], picks: IndexPicks | null): Set<string> {
  const selected = new Set<string>()
  for (const index of catalog) {
    if (index.status !== "ready") continue
    if (picks && Object.prototype.hasOwnProperty.call(picks.known, index.id)) {
      if (picks.known[index.id]) selected.add(index.id)
    } else if (index.kind === "moments") {
      selected.add(index.id)
    }
  }
  return selected
}

function hasQuotes(text: string): boolean {
  return /@\[[^\]]+?\s+\d+:\d{2}(?::\d{2})?-\d+:\d{2}(?::\d{2})?\]/.test(text)
}

function quoteToken(segment: Segment): string {
  const flat = segment.text.replace(/\s+/g, " ").replace(/[\[\]]/g, "").trim()
  const preview = flat.length <= 48 ? flat : `${flat.slice(0, 48).trimEnd()}...`
  return `@[${preview} ${clock(segment.start)}-${clock(segment.end)}]`
}

function wordIndexAt(words: Segment["words"], time: number): number {
  let found = -1
  for (let index = 0; index < words.length; index += 1) {
    if (words[index].start <= time + 0.03) found = index
    else break
  }
  return found
}

function sentenceEnd(text: string): boolean {
  return /[.!?…]["»')\]]*$/.test(text.trim())
}

function currentSentence(segment: Segment, time: number): string {
  const words = segment.words ?? []
  if (words.length === 0) {
    const parts = segment.text
      .split(/(?<=[.!?…])\s+/)
      .map((part) => part.trim())
      .filter(Boolean)
    if (parts.length <= 1) return parts[0] ?? ""
    const span = Math.max(segment.end - segment.start, 0.001)
    const ratio = Math.min(1, Math.max(0, (time - segment.start) / span))
    return parts[Math.min(parts.length - 1, Math.floor(ratio * parts.length))]
  }
  let index = wordIndexAt(words, time)
  if (index < 0) index = 0
  let start = index
  while (start > 0 && !sentenceEnd(words[start - 1].text)) start -= 1
  let end = index
  while (end < words.length - 1 && !sentenceEnd(words[end].text)) end += 1
  return words.slice(start, end + 1).map((word) => word.text).join(" ")
}

function segmentIndexAt(segments: Segment[], time: number): number {
  let found = -1
  for (let index = 0; index < segments.length; index += 1) {
    if (segments[index].start <= time + 0.15) found = index
    else break
  }
  return found
}

function FitCaption({ text }: { text: string }) {
  const ref = useRef<HTMLParagraphElement>(null)
  useEffect(() => {
    const node = ref.current
    const frame = node?.parentElement
    if (!node || !frame) return
    const limit = Math.max(36, Math.min(frame.clientHeight * 0.22, 72))
    let size = 18
    node.style.fontSize = `${size}px`
    while (size > 11 && node.scrollHeight > limit) {
      size -= 1
      node.style.fontSize = `${size}px`
    }
  }, [text])
  return (
    <p ref={ref} className="caption">
      {text}
    </p>
  )
}

function IndexBar({
  catalog,
  linesOn,
  chosen,
  busy,
  onLines,
  onToggle,
  onMoments,
  onCreate,
  onDelete,
  quoted,
}: {
  catalog: SearchIndex[]
  linesOn: boolean
  chosen: Set<string>
  busy: boolean
  onLines: () => void
  onToggle: (indexId: string, on: boolean) => void
  onMoments: () => void
  onCreate: () => void
  onDelete: (index: SearchIndex) => void
  quoted: boolean
}) {
  const menuRef = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const moments = catalog.find((index) => index.kind === "moments")
  const custom = catalog.filter((index) => index.kind === "custom")
  useEffect(() => {
    if (!open) return
    function close(event: PointerEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener("pointerdown", close)
    return () => document.removeEventListener("pointerdown", close)
  }, [open])
  return (
    <div className="index-bar">
      <label className="switch">
        <input type="checkbox" checked={linesOn} onChange={onLines} />
        Реплики
      </label>
      {moments ? (
        <span className="index-row">
          <label className="switch">
            <input
              type="checkbox"
              checked={chosen.has(moments.id)}
              disabled={moments.status !== "ready"}
              onChange={() => onToggle(moments.id, chosen.has(moments.id))}
            />
            {moments.name}
          </label>
          {moments.status === "building" && <span className="muted">строится</span>}
          {moments.error && <span className="bad">{moments.error}</span>}
        </span>
      ) : (
        <span className="index-row">
          <span>Ключевые моменты</span>
          <button type="button" className="ghost" disabled={busy} onClick={onMoments}>
            Построить
          </button>
        </span>
      )}
      <div className="index-menu" ref={menuRef}>
        <button type="button" className="index-menu-toggle" onClick={() => setOpen((value) => !value)}>
          Свои индексы
        </button>
        {open && (
          <div className="index-menu-list">
            {custom.length === 0 && <p className="muted">Пока нет.</p>}
            {custom.map((index) => (
              <span className="index-row" key={index.id}>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={chosen.has(index.id)}
                    disabled={index.status !== "ready"}
                    onChange={() => onToggle(index.id, chosen.has(index.id))}
                  />
                  {index.name}
                  {index.kind === "custom" ? (
                    <span className="index-mode">{index.strict ? "строгий" : "мягкий"}</span>
                  ) : null}
                </label>
                {index.status === "building" && <span className="muted">строится</span>}
                {index.status !== "building" && (
                  <button
                    type="button"
                    className="ghost"
                    disabled={busy}
                    onClick={() => {
                      setOpen(false)
                      onDelete(index)
                    }}
                  >
                    Удалить
                  </button>
                )}
                {index.error && <span className="bad">{index.error}</span>}
              </span>
            ))}
            <button
              type="button"
              className="tool-btn"
              disabled={busy}
              onClick={() => {
                setOpen(false)
                onCreate()
              }}
            >
              Новый индекс
            </button>
          </div>
        )}
      </div>
      {!linesOn && chosen.size === 0 && !quoted && <span className="bad">Включите индекс</span>}
    </div>
  )
}

export function Watch() {
  const { id = "" } = useParams()
  const navigate = useNavigate()
  const videoRef = useRef<HTMLMediaElement | null>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const composerRef = useRef<HTMLTextAreaElement>(null)
  const itemRefs = useRef(new Map<number, HTMLElement>())
  const dragging = useRef(false)
  const savedNames = useRef<Record<string, string>>({})

  const [detail, setDetail] = useState<Detail | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [names, setNames] = useState<Record<string, string>>({})
  const [chat, setChat] = useState<ChatMessage[]>([])
  const [agent, setAgent] = useState<AgentMessage[]>([])
  const [mode, setMode] = useState<"search" | "agent">(() =>
    localStorage.getItem(`va-mode:${id}`) === "search" ? "search" : "agent",
  )
  const [agentVersion, setAgentVersion] = useState(storedAgentVersion)
  const [draft, setDraft] = useState("")
  const [mention, setMention] = useState<{ start: number; query: string; index: number } | null>(null)
  const [sending, setSending] = useState(false)
  const [chatPending, setChatPending] = useState(false)
  const [agentPending, setAgentPending] = useState(false)
  const [voice, setVoice] = useState<AnswerVoice>(storedVoice)
  const [speechVolume, setSpeechVolume] = useState(storedSpeechVolume)
  const [chatError, setChatError] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [holdError, setHoldError] = useState<string | null>(null)
  const [dropAsk, setDropAsk] = useState(false)
  const [sync, setSync] = useState(() => localStorage.getItem("va-sync") !== "0")
  const [active, setActive] = useState(-1)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [volume, setVolume] = useState(readVolume)
  const heldVolume = useRef(readVolume() || 1)
  const [showVideo, setShowVideo] = useState(() => localStorage.getItem("va-video") !== "0")
  const [captions, setCaptions] = useState(() => localStorage.getItem("va-captions") === "1")
  const [cinema, setCinema] = useState(
    () => localStorage.getItem("va-cinema") === "1" && localStorage.getItem("va-video") !== "0",
  )
  const ticking = !detail || (detail.video.status !== "ready" && detail.video.status !== "error")
  const now = useNow(ticking)
  const [speakersOpen, setSpeakersOpen] = useState(false)
  const [catalog, setCatalog] = useState<SearchIndex[]>([])
  const [picks, setPicks] = useState<IndexPicks | null>(null)
  const [indexDialog, setIndexDialog] = useState<
    | { kind: "create"; name: string; instruction: string; strict: boolean; error?: string }
    | { kind: "delete"; index: SearchIndex; error?: string }
    | null
  >(null)
  const [indexBusy, setIndexBusy] = useState(false)
  const [naming, setNaming] = useState(false)
  const [nameError, setNameError] = useState<string | null>(null)

  useEffect(() => {
    let cancel = false
    let timer = 0
    const tick = () => {
      getVideo(id)
        .then((next) => {
          if (cancel) return
          setDetail(next)
          setLoadError(null)
          if (next.video.status !== "ready" && next.video.status !== "error") {
            timer = window.setTimeout(tick, 2000)
          }
        })
        .catch((error: unknown) => {
          if (!cancel) setLoadError(error instanceof Error ? error.message : "Ролик не загрузился")
        })
    }
    tick()
    return () => {
      cancel = true
      window.clearTimeout(timer)
    }
  }, [id])

  useEffect(() => {
    if (!detail || detail.video.status !== "ready") return
    setSegments(detail.segments)
    setChat(detail.chat)
    setAgent(detail.agent ?? [])
    setChatPending(Boolean(detail.chatPending))
    setAgentPending(Boolean(detail.agentPending))
    const initial = Object.fromEntries(detail.speakers.map((speaker) => [speaker.label, speaker.name]))
    savedNames.current = initial
    setNames(initial)
    setCatalog(detail.indexes ?? [])
    document.title = detail.video.title
  }, [detail])

  useEffect(() => {
    setPicks(readPicks(id))
    setMode(localStorage.getItem(`va-mode:${id}`) === "search" ? "search" : "agent")
    stopSpeech()
    return () => {
      stopSpeech()
    }
  }, [id])

  useEffect(() => {
    if (!id || (!chatPending && !agentPending)) return
    let cancel = false
    let timer = 0
    const tick = () => {
      getDialog(id)
        .then((next) => {
          if (cancel) return
          setChat(next.chat)
          setAgent(next.agent)
          setChatPending(next.chatPending)
          setAgentPending(next.agentPending)
          setDetail((current) =>
            current
              ? {
                  ...current,
                  chat: next.chat,
                  agent: next.agent,
                  chatPending: next.chatPending,
                  agentPending: next.agentPending,
                }
              : current,
          )
          if (next.chatPending || next.agentPending) timer = window.setTimeout(tick, 2000)
        })
        .catch(() => {
          if (!cancel) timer = window.setTimeout(tick, 2000)
        })
    }
    timer = window.setTimeout(tick, 1500)
    return () => {
      cancel = true
      window.clearTimeout(timer)
    }
  }, [id, chatPending, agentPending])

  const building = catalog.some((index) => index.status === "building")
  useEffect(() => {
    if (!id || !building) return
    let cancel = false
    const timer = window.setInterval(() => {
      listIndexes(id)
        .then((next) => {
          if (!cancel) setCatalog(next.indexes)
        })
        .catch(() => {})
    }, 2000)
    return () => {
      cancel = true
      window.clearInterval(timer)
    }
  }, [id, building])

  useEffect(() => {
    if (!sync || active < 0) return
    const container = listRef.current
    const item = itemRefs.current.get(active)
    if (!container || !item) return
    const box = container.getBoundingClientRect()
    const row = item.getBoundingClientRect()
    if (row.top < box.top || row.bottom > box.bottom) {
      item.scrollIntoView({ block: "center", behavior: "smooth" })
    }
  }, [active, sync])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ block: "end" })
  }, [chat.length, sending])

  useEffect(() => {
    const node = videoRef.current
    if (!node) return
    node.volume = volume
    node.muted = volume === 0
  }, [volume])

  function changeVolume(next: number) {
    const level = Math.min(1, Math.max(0, next))
    if (level > 0) heldVolume.current = level
    setVolume(level)
    localStorage.setItem("va-volume", String(level))
  }

  function seek(seconds: number) {
    const node = videoRef.current
    if (!node) return
    node.currentTime = seconds
    setTime(seconds)
    setActive(segmentIndexAt(segments, seconds))
  }

  function onTime() {
    const node = videoRef.current
    if (!node || dragging.current) return
    setTime(node.currentTime)
    if (sync) setActive(segmentIndexAt(segments, node.currentTime))
  }

  function toggleSync(next: boolean) {
    setSync(next)
    localStorage.setItem("va-sync", next ? "1" : "0")
    if (next && videoRef.current) setActive(segmentIndexAt(segments, videoRef.current.currentTime))
  }

  function closeSpeakers() {
    setNames({ ...savedNames.current })
    setNameError(null)
    setSpeakersOpen(false)
  }

  async function saveNames() {
    if (!detail || naming) return
    const jobs = detail.speakers.flatMap((speaker) => {
      const name = (names[speaker.label] ?? "").trim()
      const previous = savedNames.current[speaker.label] ?? speaker.label
      if (!name || name === previous) return []
      return [{ label: speaker.label, name }]
    })
    setNames((current) => {
      const next = { ...current }
      for (const speaker of detail.speakers) {
        if (!(current[speaker.label] ?? "").trim()) {
          next[speaker.label] = savedNames.current[speaker.label] ?? speaker.label
        }
      }
      return next
    })
    if (jobs.length === 0) {
      setSpeakersOpen(false)
      return
    }
    setNaming(true)
    setNameError(null)
    try {
      let latest = savedNames.current
      for (const job of jobs) {
        const updated = await renameSpeaker(id, job.label, job.name)
        latest = Object.fromEntries(updated.speakers.map((speaker) => [speaker.label, speaker.name]))
        savedNames.current = latest
      }
      setNames(latest)
      setSegments((current) =>
        current.map((segment) => {
          const name = segment.speaker ? latest[segment.speaker] : undefined
          return name ? { ...segment, speakerName: name } : segment
        }),
      )
      setSpeakersOpen(false)
    } catch (error: unknown) {
      setNameError(error instanceof Error ? error.message : "Имя не сохранилось")
    } finally {
      setNaming(false)
    }
  }

  useEffect(() => {
    const node = composerRef.current
    if (!node) return
    node.style.height = "auto"
    node.style.height = `${Math.min(node.scrollHeight, 180)}px`
  }, [draft])

  function refreshMention(value: string, cursor: number) {
    const found = mentionAt(value, cursor)
    if (!found) {
      setMention(null)
      return
    }
    setMention((current) =>
      current && current.start === found.start && current.query === found.query ? current : { ...found, index: 0 },
    )
  }

  function pickMention(name: string) {
    if (!mention) return
    const node = composerRef.current
    const cursor = node?.selectionStart ?? draft.length
    const next = `${draft.slice(0, mention.start)}@${name} ${draft.slice(cursor)}`
    const caret = mention.start + name.length + 2
    setDraft(next)
    setMention(null)
    requestAnimationFrame(() => {
      node?.focus()
      node?.setSelectionRange(caret, caret)
    })
  }

  function remember(next: IndexPicks) {
    setPicks(next)
    localStorage.setItem(`va-indexes:${id}`, JSON.stringify(next))
  }

  function toggleLines() {
    const lines = !(picks?.lines ?? true)
    remember({ lines, known: picks?.known ?? {} })
  }

  function toggleIndex(indexId: string, on: boolean) {
    remember({ lines: picks?.lines ?? true, known: { ...(picks?.known ?? {}), [indexId]: !on } })
  }

  async function refreshIndexes() {
    const next = await listIndexes(id)
    setCatalog(next.indexes)
  }

  async function runIndex(action: () => Promise<SearchIndex>) {
    setIndexBusy(true)
    setChatError(null)
    try {
      await action()
      await refreshIndexes()
    } catch (error: unknown) {
      setChatError(error instanceof Error ? error.message : "Индекс не запустился")
    } finally {
      setIndexBusy(false)
    }
  }

  async function confirmIndex() {
    if (!indexDialog || indexDialog.kind !== "create" || indexBusy) return
    const name = indexDialog.name.trim()
    const instruction = indexDialog.instruction.trim()
    if (!name || !instruction) {
      setIndexDialog({ ...indexDialog, error: "Нужны название и инструкция" })
      return
    }
    setIndexBusy(true)
    try {
      await createIndex(id, name, instruction, indexDialog.strict)
      await refreshIndexes()
      setIndexDialog(null)
    } catch (error: unknown) {
      setIndexDialog({
        ...indexDialog,
        error: error instanceof Error ? error.message : "Индекс не создался",
      })
    } finally {
      setIndexBusy(false)
    }
  }

  async function confirmDrop() {
    if (!indexDialog || indexDialog.kind !== "delete" || indexBusy) return
    setIndexBusy(true)
    try {
      await removeIndex(id, indexDialog.index.id)
      await refreshIndexes()
      setIndexDialog(null)
    } catch (error: unknown) {
      setIndexDialog({
        ...indexDialog,
        error: error instanceof Error ? error.message : "Индекс не удалился",
      })
    } finally {
      setIndexBusy(false)
    }
  }

  function quoteSegment(segment: Segment) {
    const token = quoteToken(segment)
    setDraft((current) => {
      const gap = current && !/\s$/.test(current) ? " " : ""
      return `${current}${gap}${token} `
    })
    setChatError(null)
    window.setTimeout(() => {
      const node = composerRef.current
      if (!node) return
      node.focus()
      const end = node.value.length
      node.setSelectionRange(end, end)
    }, 0)
  }

  function downloadDialogue() {
    if (!detail) return
    const named = detail.video.diarize !== false
    const payload = {
      title: detail.video.title,
      messages: segments.map((segment) => ({
        ...(named ? { speaker: segment.speakerName } : {}),
        start: segment.start,
        end: segment.end,
        text: segment.text,
      })),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    const name = detail.video.title.replace(/[^\p{L}\p{N}]+/gu, " ").trim().replace(/\s+/g, "-") || "dialog"
    link.href = url
    link.download = `${name}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  async function clearDialog() {
    if (sending) return
    setChatError(null)
    try {
      if (mode === "agent") {
        await clearAgent(id)
        setAgent([])
        setAgentPending(false)
        setDetail((current) => (current ? { ...current, agent: [], agentPending: false } : current))
      } else {
        await clearChat(id)
        setChat([])
        setChatPending(false)
        setDetail((current) => (current ? { ...current, chat: [], chatPending: false } : current))
      }
    } catch (error: unknown) {
      setChatError(error instanceof Error ? error.message : "Диалог не очистился")
    }
  }

  async function stopTask() {
    const kind = mode === "agent" ? "agent" : "chat"
    setChatError(null)
    try {
      const pending = await stopDialog(id, kind)
      const next = await getDialog(id)
      setChat(next.chat)
      setAgent(next.agent)
      setChatPending(pending.chatPending)
      setAgentPending(pending.agentPending)
      setDetail((current) =>
        current
          ? {
              ...current,
              chat: next.chat,
              agent: next.agent,
              chatPending: pending.chatPending,
              agentPending: pending.agentPending,
            }
          : current,
      )
    } catch (error: unknown) {
      setChatError(error instanceof Error ? error.message : "Не удалось остановить")
    }
  }

  async function sendQuestion() {
    const question = draft.trim()
    const linesOn = picks?.lines ?? true
    const indexIds = [...chosenIds(catalog, picks)]
    if (!question || sending) return
    if (mode === "agent" ? agentPending : chatPending) return
    if (mode === "search" && !linesOn && indexIds.length === 0 && !hasQuotes(question)) {
      setChatError("Включите хотя бы один индекс")
      return
    }
    setDraft("")
    setMention(null)
    setSending(true)
    setChatError(null)
    if (mode === "agent") {
      setAgent((current) => [...current, { role: "user", content: question, actions: [], citations: [] }])
      try {
        await askAgent(id, question, agentVersion)
        setAgentPending(true)
      } catch (error: unknown) {
        setAgent((current) => current.slice(0, -1))
        setChatError(error instanceof Error ? error.message : "Ответ не получен")
      } finally {
        setSending(false)
      }
      return
    }
    setChat((current) => [...current, { role: "user", content: question, citations: [] }])
    try {
      await askVideo(id, question, linesOn, indexIds)
      setChatPending(true)
    } catch (error: unknown) {
      setChat((current) => current.slice(0, -1))
      setChatError(error instanceof Error ? error.message : "Ответ не получен")
    } finally {
      setSending(false)
    }
  }

  if (loadError) {
    return (
      <main className="pending">
        <Link className="back" to="/">К списку</Link>
        <h1>Ролик не открылся</h1>
        <p className="bad">{loadError}</p>
      </main>
    )
  }

  async function toggleHold() {
    if (!detail) return
    setHoldError(null)
    try {
      const updated = await holdVideo(id, !detail.video.held)
      setDetail((current) => (current ? { ...current, video: updated } : current))
    } catch (error: unknown) {
      setHoldError(error instanceof Error ? error.message : "Не удалось поставить на паузу")
    }
  }

  async function confirmDeleteVideo() {
    setHoldError(null)
    try {
      await removeVideo(id)
      navigate("/")
    } catch (error: unknown) {
      setHoldError(error instanceof Error ? error.message : "Не удалось удалить ролик")
      setDropAsk(false)
    }
  }

  if (!detail || (detail.video.status !== "ready" && detail.video.status !== "error")) {
    return (
      <main className="pending">
        <Link className="back" to="/">К списку</Link>
        <h1>{detail?.video.title ?? "Ролик"}</h1>
        <p className="stage-label" aria-live="polite">
          {detail
            ? `${statusText(detail.video)}${stageMark(detail.video)}${runningClock(detail.video, now) ? ` · ${runningClock(detail.video, now)}` : ""}`
            : "Загрузка"}
        </p>
        {detail && (
          <div className={detail.video.held ? "meter" : "meter live"} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={detail.video.progress}>
            <span style={{ width: `${detail.video.progress}%` }} />
          </div>
        )}
        {detail && (
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => void toggleHold()}>
              {detail.video.held ? "Продолжить" : "Пауза"}
            </button>
            <button type="button" className="ghost" onClick={() => setDropAsk(true)}>
              Удалить
            </button>
          </div>
        )}
        {holdError && <p className="bad">{holdError}</p>}
        {dropAsk && detail && (
          <Modal title="Удалить ролик?" onClose={() => setDropAsk(false)}>
            <p>Вместе с «{detail.video.title}» пропадут незаконченная обработка и файлы.</p>
            <div className="modal-actions">
              <button type="button" className="ghost" onClick={() => setDropAsk(false)}>Отмена</button>
              <button type="button" className="danger" onClick={() => void confirmDeleteVideo()}>Удалить</button>
            </div>
          </Modal>
        )}
      </main>
    )
  }

  if (detail.video.status === "error") {
    return (
      <main className="pending">
        <Link className="back" to="/">К списку</Link>
        <h1>{detail.video.title}</h1>
        <p className="bad">{detail.video.error || "Разбор не удался"}</p>
      </main>
    )
  }

  const mediaDuration = duration || detail.video.durationSec || 0
  const audioOnly = detail.video.audio
  const linesOn = picks?.lines ?? true
  const chosen = chosenIds(catalog, picks)
  const taskBusy = mode === "agent" ? agentPending || sending : chatPending || sending
  const named = detail.video.diarize !== false
  const speakerList: Speaker[] = named
    ? detail.speakers.map((speaker) => ({
        label: speaker.label,
        name: (names[speaker.label] ?? speaker.name).trim() || speaker.label,
      }))
    : []
  const mentionOptions = mention
    ? speakerList.filter((speaker) => speaker.name.toLocaleLowerCase("ru").startsWith(mention.query.toLocaleLowerCase("ru")))
    : []
  const mentionIndex = mention ? Math.min(mention.index, Math.max(mentionOptions.length - 1, 0)) : 0
  const liveActions = agent[agent.length - 1]?.role === "assistant" ? agent[agent.length - 1].actions : []
  const liveStatus = liveActionStatus(liveActions)
  const liveProgress = liveActions[liveActions.length - 1]?.progress

  return (
    <main className="studio">
      <header className="top">
        <div className="top-side">
          <Link className="back" to="/">К списку</Link>
          <button type="button" className="tool-btn" onClick={downloadDialogue}>
            Выгрузить диалог
          </button>
          {named && (
            <button type="button" className="tool-btn" onClick={() => setSpeakersOpen(true)}>
              Имена спикеров
            </button>
          )}
        </div>
        <h1 title={detail.video.title}>{detail.video.title}</h1>
        <div className="tools">
          {detail.video.sourceUrl && (
            <a
              className="tool-btn"
              href={originalUrl(detail.video.sourceUrl, time)}
              target="_blank"
              rel="noreferrer"
            >
              Оригинал
            </a>
          )}
          <label className="switch">
            <input type="checkbox" checked={sync} onChange={(event) => toggleSync(event.target.checked)} />
            Синхронизация
          </label>
        </div>
        {detail.video.warning && <p className="warn">{detail.video.warning}</p>}
      </header>

      <Panes
        showVideo={!audioOnly && showVideo}
        cinema={!audioOnly && cinema}
        dialog={
        <section className="panel">
          <div className="panel-head">Диалог</div>
          <div className="transcript" ref={listRef}>
            {segments.length === 0 && <p className="muted empty">Реплик нет.</p>}
            {segments.map((segment, index) => {
              const words = segment.words ?? []
              const playing = wordIndexAt(words, time)
              const here = index === segmentIndexAt(segments, time)
              return (
              <div
                key={segment.id}
                className={index === active ? "message active" : "message"}
                ref={(node) => {
                  if (node) itemRefs.current.set(index, node)
                  else itemRefs.current.delete(index)
                }}
                onClick={() => {
                  if (sync) seek(segment.start)
                }}
              >
                <span className="time">{clock(segment.start)}</span>
                <span>
                  {named && segment.speakerName ? (
                    <span className="who" style={{ color: `hsl(${speakerHue(segment.speaker ?? segment.speakerName)} var(--who-s) var(--who-l))` }}>
                      {segment.speakerName}
                    </span>
                  ) : null}
                  <button
                    type="button"
                    className="reply"
                    title="Искать по реплике"
                    aria-label="Искать по реплике"
                    onClick={(event) => {
                      event.stopPropagation()
                      quoteSegment(segment)
                    }}
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path d="M6.2 4.2 2.8 7.6l3.4 3.4" />
                      <path d="M3.2 7.6H9a3.4 3.4 0 0 1 3.4 3.4V13" />
                    </svg>
                  </button>
                  <CopyButton text={segment.text} />
                  <p>
                    {words.length > 0
                      ? words.map((word, wordIndex) => (
                          <span
                            key={`${segment.id}-${wordIndex}`}
                            className={here && wordIndex === playing ? "word on" : "word"}
                            onClick={(event) => {
                              event.stopPropagation()
                              if (sync) seek(word.start)
                            }}
                          >
                            {word.text}
                            {wordIndex < words.length - 1 ? " " : ""}
                          </span>
                        ))
                      : (
                          <span
                            className="word"
                            onClick={(event) => {
                              event.stopPropagation()
                              if (sync) seek(segment.start)
                            }}
                          >
                            {segment.text}
                          </span>
                        )}
                  </p>
                </span>
              </div>
              )
            })}
          </div>
        </section>
        }
        video={
        audioOnly ? (
          <audio
            ref={(node) => {
              videoRef.current = node
            }}
            src={`/api/videos/${id}/media`}
            preload="metadata"
            onLoadedMetadata={(event) => {
              const node = event.currentTarget
              node.volume = volume
              node.muted = volume === 0
              setDuration(node.duration || 0)
            }}
            onTimeUpdate={onTime}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        ) : (
        <section className="panel video-panel">
          <div className="panel-head">
            Ролик
            <button
              type="button"
              className={captions ? "on" : ""}
              aria-pressed={captions}
              onClick={() => {
                const next = !captions
                localStorage.setItem("va-captions", next ? "1" : "0")
                setCaptions(next)
              }}
            >
              Субтитры
            </button>
          </div>
          <div className="screen">
          <video
            ref={(node) => {
              videoRef.current = node
            }}
            src={`/api/videos/${id}/media`}
            preload="metadata"
            playsInline
            onLoadedMetadata={(event) => {
              const node = event.currentTarget
              node.volume = volume
              node.muted = volume === 0
              setDuration(node.duration || 0)
            }}
            onClick={() => {
              const node = videoRef.current
              if (!node) return
              if (node.paused) void node.play()
              else node.pause()
            }}
            onTimeUpdate={onTime}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
          {captions && active >= 0 && segments[active] && time <= segments[active].end + 0.25 && (
            <FitCaption text={currentSentence(segments[active], time)} />
          )}
          </div>
        </section>
        )
        }
        chat={
        <section className="panel">
          <div className="panel-head">
            <div className="modes">
              <button
                type="button"
                className={mode === "agent" ? "on" : ""}
                title="Сам решает, чем ответить на задачу"
                onClick={() => {
                  setMode("agent")
                  localStorage.setItem(`va-mode:${id}`, "agent")
                }}
              >
                Агент
              </button>
              <button
                type="button"
                className={mode === "search" ? "on" : ""}
                title="Векторный поиск по репликам и индексам"
                onClick={() => {
                  setMode("search")
                  localStorage.setItem(`va-mode:${id}`, "search")
                }}
              >
                Поиск
              </button>
            </div>
            <div className="voice-set">
              <button
                type="button"
                className={voice === "m" ? "on" : ""}
                aria-pressed={voice === "m"}
                onClick={() => {
                  localStorage.setItem("va-voice", "m")
                  setVoice("m")
                }}
              >
                Мужской
              </button>
              <button
                type="button"
                className={voice === "f" ? "on" : ""}
                aria-pressed={voice === "f"}
                onClick={() => {
                  localStorage.setItem("va-voice", "f")
                  setVoice("f")
                }}
              >
                Женский
              </button>
              <label className="voice-volume" title="Громкость голоса">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={speechVolume}
                  aria-label="Громкость голоса"
                  onChange={(event) => {
                    const next = Number(event.target.value)
                    localStorage.setItem("va-voice-volume", String(next))
                    setSpeechVolume(next)
                    applySpeechVolume(next)
                  }}
                />
              </label>
            </div>
            <button
              type="button"
              className="clear-chat"
              aria-label="Очистить"
              title="Очистить"
              disabled={(mode === "agent" ? agent : chat).length === 0 || sending}
              onClick={() => void clearDialog()}
            >
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M4 4 12 12M12 4 4 12" />
              </svg>
            </button>
          </div>
          <div className="chat-log">
            {(mode === "agent" ? agent : chat).length === 0 && (
              <p className="muted empty">
                {mode === "agent"
                  ? "Поставьте задачу по этому ролику."
                  : "Найдите, когда прозвучала мысль или о чём была речь."}
              </p>
            )}
            {(mode === "agent" ? agent : chat).map((message, index) => (
              <article key={`${mode}-${message.role}-${index}`} className={`bubble ${message.role}`}>
                {message.role === "assistant" && (
                  <>
                    <SpeakButton text={message.content} token={`${mode}-${index}`} voice={voice} />
                    <CopyButton text={message.content} />
                  </>
                )}
                <MessageText
                  text={message.content}
                  speakers={speakerList}
                  onSeek={(seconds) => {
                    if (sync) seek(seconds)
                  }}
                />
                {mode === "agent" && (message as AgentMessage).actions.length > 0 && (
                  <details className="sources">
                    <summary>Actions · {(message as AgentMessage).actions.length}</summary>
                    <div className="source-list">
                      {(message as AgentMessage).actions.map((action, step) => {
                        const hits = action.hits ?? []
                        const wide = hits.length > 0 || action.detail.length > 80
                        return (
                          <div key={`${action.title}-${step}`} className="step">
                            <div className="step-line">
                              <span className="kind">{action.title}</span>
                              {action.detail ? (
                                <span className="step-query">
                                  <MentionBits text={action.detail} speakers={speakerList} />
                                </span>
                              ) : null}
                            </div>
                            {typeof action.progress === "number" && (
                              <div
                                className="step-bar"
                                role="progressbar"
                                aria-label="Сборка индекса"
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-valuenow={Math.round(action.progress * 100)}
                              >
                                <div style={{ width: `${Math.min(100, Math.max(0, action.progress * 100))}%` }} />
                              </div>
                            )}
                            {wide && (
                              <details className="step-more">
                                <summary>Подробнее</summary>
                                {action.detail.length > 80 && (
                                  <p className="step-text">
                                    <MentionBits text={action.detail} speakers={speakerList} />
                                  </p>
                                )}
                                {hits.map((hit) => (
                                  <button
                                    key={`${hit.start}-${hit.text.slice(0, 24)}`}
                                    type="button"
                                    className="hit"
                                    onClick={() => {
                                      if (sync) seek(hit.start)
                                    }}
                                  >
                                    <span className="hit-meta">
                                      {clock(hit.start)}
                                      {named && hit.speakerName ? (
                                        <>
                                          {" "}
                                          <MentionBits text={hit.speakerName} speakers={speakerList} />
                                        </>
                                      ) : null}
                                    </span>
                                    <span className="hit-text">
                                      <MentionBits text={hit.text} speakers={speakerList} />
                                    </span>
                                  </button>
                                ))}
                              </details>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </details>
                )}
                {mode === "search" && message.citations.length > 0 && (
                  <details className="sources">
                    <summary>Источники · {message.citations.length}</summary>
                    <div className="source-list">
                      {message.citations.map((citation) => (
                        <button
                          key={`${citation.type ?? "m"}-${citation.segmentId}-${citation.start}`}
                          type="button"
                          className="source"
                          onClick={() => {
                            if (sync) seek(citation.start)
                          }}
                        >
                          <span className="kind">{citation.type || "m"}</span>
                          <span className="time">{clock(citation.start)}</span>
                          <span>
                            {named && citation.speakerName ? <span className="who">{citation.speakerName}</span> : null}
                            <p>{citation.text}</p>
                          </span>
                        </button>
                      ))}
                    </div>
                  </details>
                )}
              </article>
            ))}
            {mode === "search" && (chatPending || sending) && <p className="muted">Ищу по ролику…</p>}
            {mode === "agent" && (agentPending || sending) && liveStatus && (
              <p className="muted live-status">
                <MentionBits text={liveStatus} speakers={speakerList} />
                {typeof liveProgress === "number" && (
                  <span
                    className="step-bar"
                    role="progressbar"
                    aria-label="Сборка индекса"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(liveProgress * 100)}
                  >
                    <span style={{ width: `${Math.min(100, Math.max(0, liveProgress * 100))}%` }} />
                  </span>
                )}
              </p>
            )}
            {chatError && <p className="bad">{chatError}</p>}
            <div ref={chatEndRef} />
          </div>
          {mode === "search" && <IndexBar
            catalog={catalog}
            linesOn={linesOn}
            chosen={chosen}
            busy={indexBusy}
            onLines={toggleLines}
            onToggle={toggleIndex}
            onMoments={() => void runIndex(() => buildMoments(id))}
            onCreate={() => setIndexDialog({ kind: "create", name: "", instruction: "", strict: true })}
            onDelete={(index) => setIndexDialog({ kind: "delete", index })}
            quoted={hasQuotes(draft)}
          />}
          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault()
              if (taskBusy) return
              void sendQuestion()
            }}
          >
            {mention && mentionOptions.length > 0 && (
              <ul className="mention-list" role="listbox">
                {mentionOptions.map((speaker, index) => (
                  <li key={speaker.label}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={index === mentionIndex}
                      className={index === mentionIndex ? "active" : ""}
                      onMouseDown={(event) => {
                        event.preventDefault()
                        pickMention(speaker.name)
                      }}
                    >
                      @{speaker.name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            <textarea
              ref={composerRef}
              value={draft}
              rows={2}
              disabled={taskBusy}
              placeholder={mode === "agent" ? "Задача" : "Поиск, @имя спикера"}
              onChange={(event) => {
                setDraft(event.target.value)
                refreshMention(event.target.value, event.target.selectionStart ?? event.target.value.length)
              }}
              onClick={(event) => refreshMention(event.currentTarget.value, event.currentTarget.selectionStart ?? 0)}
              onKeyUp={(event) => {
                if (event.key === "ArrowUp" || event.key === "ArrowDown" || event.key === "Enter" || event.key === "Escape") return
                refreshMention(event.currentTarget.value, event.currentTarget.selectionStart ?? 0)
              }}
              onKeyDown={(event) => {
                if (mention && mentionOptions.length > 0) {
                  if (event.key === "ArrowDown") {
                    event.preventDefault()
                    setMention((current) =>
                      current ? { ...current, index: (mentionIndex + 1) % mentionOptions.length } : current,
                    )
                    return
                  }
                  if (event.key === "ArrowUp") {
                    event.preventDefault()
                    setMention((current) =>
                      current
                        ? { ...current, index: (mentionIndex - 1 + mentionOptions.length) % mentionOptions.length }
                        : current,
                    )
                    return
                  }
                  if (event.key === "Enter" || event.key === "Tab") {
                    event.preventDefault()
                    pickMention(mentionOptions[mentionIndex].name)
                    return
                  }
                  if (event.key === "Escape") {
                    event.preventDefault()
                    setMention(null)
                    return
                  }
                }
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault()
                  if (!taskBusy) void sendQuestion()
                }
              }}
            />
            {taskBusy ? (
              <button type="button" className="send stop" aria-label="Остановить" title="Остановить" onClick={() => void stopTask()}>
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  <rect x="7" y="7" width="10" height="10" rx="1.5" />
                </svg>
              </button>
            ) : (
              <button
                type="submit"
                className="send"
                aria-label="Отправить"
                title="Отправить"
                disabled={!draft.trim() || (mode === "search" && !linesOn && chosen.size === 0 && !hasQuotes(draft))}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M1.8 8.2 14 2.4 8.6 14.2 7.2 9.4 1.8 8.2z" />
                </svg>
              </button>
            )}
            {mode === "agent" && (
              <div className="agent-version" role="radiogroup" aria-label="Версия агента">
                {AGENT_VERSIONS.map((version) => (
                  <button
                    key={version.id}
                    type="button"
                    role="radio"
                    aria-checked={agentVersion === version.id}
                    className={agentVersion === version.id ? "on" : ""}
                    onClick={() => {
                      localStorage.setItem("va-agent-version", version.id)
                      setAgentVersion(version.id)
                    }}
                  >
                    {version.label}
                  </button>
                ))}
              </div>
            )}
          </form>
        </section>
        }
      />
      <footer className={audioOnly ? "transport audio" : "transport"}>
        <button
          type="button"
          className="icon-btn"
          aria-label={playing ? "Пауза" : audioOnly ? "Слушать" : "Смотреть"}
          onClick={() => {
            const node = videoRef.current
            if (!node) return
            if (node.paused) void node.play()
            else node.pause()
          }}
        >
          {playing ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M7 5h3.5v14H7zm6.5 0H17v14h-3.5z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M8 5.2v13.6L18.8 12z" />
            </svg>
          )}
        </button>
        <span className="time">
          {clock(time)} / {clock(mediaDuration)}
        </span>
        <input
          type="range"
          min={0}
          max={mediaDuration || 0}
          step={0.1}
          value={Math.min(time, mediaDuration || 0)}
          aria-label="Положение"
          onPointerDown={() => {
            dragging.current = true
          }}
          onPointerUp={() => {
            dragging.current = false
          }}
          onChange={(event) => seek(Number(event.target.value))}
        />
        <input
          className="volume"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          aria-label="Громкость"
          onChange={(event) => changeVolume(Number(event.target.value))}
        />
        <button
          type="button"
          className="icon-btn"
          aria-label={volume === 0 ? "Включить звук" : "Выключить звук"}
          aria-pressed={volume === 0}
          onClick={() => changeVolume(volume === 0 ? heldVolume.current || 1 : 0)}
        >
          {volume === 0 ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 9v6h4l5 4V5L8 9H4zm12.2 3 2.3-2.3 1.4 1.4L17.6 13.4l2.3 2.3-1.4 1.4-2.3-2.3-2.3 2.3-1.4-1.4 2.3-2.3-2.3-2.3 1.4-1.4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 9v6h4l5 4V5L8 9H4zm11.5 3a3.5 3.5 0 0 0-1.8-3.1v6.2A3.5 3.5 0 0 0 15.5 12zm0-7.2v2.1a6.5 6.5 0 0 1 0 10.2v2.1a8.5 8.5 0 0 0 0-14.4z" />
            </svg>
          )}
        </button>
        {!audioOnly && <button
          type="button"
          className={showVideo ? "icon-btn" : "icon-btn pressed"}
          aria-pressed={!showVideo}
          aria-label={showVideo ? "Скрыть видео" : "Показать видео"}
          onClick={() => {
            const next = !showVideo
            localStorage.setItem("va-video", next ? "1" : "0")
            setShowVideo(next)
            if (!next) {
              localStorage.setItem("va-cinema", "0")
              setCinema(false)
            }
          }}
        >
          {showVideo ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7.5A1.5 1.5 0 0 1 5.5 6h8A1.5 1.5 0 0 1 15 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 4 16.5v-9zm12 1.8 3.2-1.8v9l-3.2-1.8v-5.4z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 7.5A1.5 1.5 0 0 1 5.5 6h8A1.5 1.5 0 0 1 15 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 4 16.5v-9zm12 1.8 3.2-1.8v9l-3.2-1.8v-5.4zM3.2 4.4 19.6 20.8l1.2-1.2L4.4 3.2 3.2 4.4z" />
            </svg>
          )}
        </button>}
        {!audioOnly && <button
          type="button"
          className={cinema ? "icon-btn pressed" : "icon-btn"}
          aria-pressed={cinema}
          aria-label={cinema ? "Убрать видео с полного экрана" : "Видео на весь экран"}
          onClick={() => {
            const next = !cinema
            localStorage.setItem("va-cinema", next ? "1" : "0")
            setCinema(next)
            if (next) {
              localStorage.setItem("va-video", "1")
              setShowVideo(true)
            }
          }}
        >
          {cinema ? (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M9 4v3H6v2h5V4H9zm4 0v5h5V7h-3V4h-2zM4 15h5v5h2v-7H4v2zm11 0v7h2v-3h3v-2h-5z" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M4 10V4h6v2H6v4H4zm10-6h6v6h-2V6h-4V4zM4 14h2v4h4v2H4v-6zm10 4v2h6v-6h-2v4h-4z" />
            </svg>
          )}
        </button>}
      </footer>
      {speakersOpen && (
        <Modal title="Имена спикеров" onClose={closeSpeakers}>
          {detail.speakers.length === 0 && <p className="muted">Спикеров нет.</p>}
          <div className="speakers">
            {detail.speakers.map((speaker) => (
              <label className="speaker" key={speaker.label}>
                <span style={{ color: `hsl(${speakerHue(speaker.label)} var(--who-s) var(--who-l))` }}>
                  {speaker.label}
                </span>
                <input
                  value={names[speaker.label] ?? speaker.name}
                  aria-label={`Имя для ${speaker.label}`}
                  onChange={(event) => setNames((current) => ({ ...current, [speaker.label]: event.target.value }))}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void saveNames()
                  }}
                />
              </label>
            ))}
          </div>
          {nameError && <p className="bad">{nameError}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={closeSpeakers}>
              Отмена
            </button>
            <button type="button" className="send" disabled={naming} onClick={() => void saveNames()}>
              Сохранить
            </button>
          </div>
        </Modal>
      )}
      {indexDialog?.kind === "create" && (
        <Modal title="Новый индекс" onClose={() => setIndexDialog(null)}>
          <input
            className="modal-field"
            value={indexDialog.name}
            autoFocus
            placeholder="Название"
            aria-label="Название индекса"
            onChange={(event) => setIndexDialog({ ...indexDialog, name: event.target.value, error: undefined })}
          />
          <textarea
            className="modal-field"
            rows={3}
            value={indexDialog.instruction}
            placeholder="Что вытащить из ролика"
            aria-label="Инструкция индекса"
            onChange={(event) =>
              setIndexDialog({ ...indexDialog, instruction: event.target.value, error: undefined })
            }
          />
          <label className="switch tip">
            <input
              type="checkbox"
              checked={indexDialog.strict}
              onChange={(event) => setIndexDialog({ ...indexDialog, strict: event.target.checked, error: undefined })}
            />
            Строгий
            <span className="usage-tip" role="tooltip">
              Строгий молчит, если в отрезке нет того, что вы просите. Снимите галочку — мягкий индекс пишет выжимку в каждом отрезке.
            </span>
          </label>
          {indexDialog.error && <p className="bad">{indexDialog.error}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => setIndexDialog(null)}>
              Отмена
            </button>
            <button type="button" className="send" disabled={indexBusy} onClick={() => void confirmIndex()}>
              Построить
            </button>
          </div>
        </Modal>
      )}
      {indexDialog?.kind === "delete" && (
        <Modal title="Удалить индекс?" onClose={() => setIndexDialog(null)}>
          <p>Пропадёт «{indexDialog.index.name}» и его пункты.</p>
          {indexDialog.error && <p className="bad">{indexDialog.error}</p>}
          <div className="modal-actions">
            <button type="button" className="ghost" onClick={() => setIndexDialog(null)}>
              Отмена
            </button>
            <button type="button" className="danger" disabled={indexBusy} onClick={() => void confirmDrop()}>
              Удалить
            </button>
          </div>
        </Modal>
      )}
    </main>
  )
}
