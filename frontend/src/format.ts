import { useEffect, useState } from "react"

export function bytes(size: number): string {
  const amount = Math.max(0, Math.round(size))
  if (amount < 1024) return `${amount} Б`
  const kilo = amount / 1024
  if (kilo < 10) return `${kilo.toFixed(1).replace(".", ",")} КБ`
  if (kilo < 1024) return `${Math.round(kilo)} КБ`
  const mega = kilo / 1024
  if (mega < 10) return `${mega.toFixed(1).replace(".", ",")} МБ`
  if (mega < 1024) return `${Math.round(mega)} МБ`
  const giga = mega / 1024
  if (giga < 10) return `${giga.toFixed(1).replace(".", ",")} ГБ`
  return `${Math.round(giga)} ГБ`
}

export function clock(seconds: number): string {
  const total = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const ss = String(secs).padStart(2, "0")
  if (hours) return `${hours}:${String(minutes).padStart(2, "0")}:${ss}`
  return `${minutes}:${ss}`
}

export const STATUS: Record<string, string> = {
  queued: "в очереди",
  fetching: "скачивание",
  compressing: "сжатие",
  storing: "сохранение",
  transcribing: "распознавание",
  diarizing: "диаризация",
  merging: "склейка фраз",
  correcting: "правка текста",
  splitting: "разбивка текста",
  embedding: "индексация",
  moments: "ключевые моменты",
  naming: "автоименование",
  ready: "готово",
  error: "ошибка",
}

export function useNow(live: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!live) return
    const timer = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(timer)
  }, [live])
  return now
}

export function runningClock(
  video: { status: string; createdAt: string; startedAt: string | null; held?: boolean },
  now: number,
): string | null {
  if (video.held || video.status === "ready" || video.status === "error") return null
  const from = video.startedAt && video.status !== "queued" ? video.startedAt : video.createdAt
  if (!from) return null
  const start = Date.parse(from)
  if (!Number.isFinite(start)) return null
  return clock(Math.max(0, (now - start) / 1000))
}

export function stageMark(video: { status: string; stageProgress: number }): string {
  if (video.status === "ready" || video.status === "error" || video.status === "queued") return ""
  return ` · ${video.stageProgress}%`
}

export function statusText(video: { status: string; queuePlace: number | null; held?: boolean }): string {
  if (video.held && video.status !== "ready" && video.status !== "error") {
    const stage = STATUS[video.status]
    return stage ? `пауза · ${stage}` : "пауза"
  }
  if (video.status === "queued" && video.queuePlace && video.queuePlace > 1) {
    return `в очереди · ${video.queuePlace}`
  }
  return STATUS[video.status] ?? video.status
}

export function originalUrl(raw: string, seconds: number): string {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    return raw
  }
  const host = url.hostname.toLowerCase().replace(/^www\./, "")
  const youtube =
    host === "youtu.be" ||
    host === "youtube.com" ||
    host.endsWith(".youtube.com") ||
    host === "youtube-nocookie.com" ||
    host.endsWith(".youtube-nocookie.com")
  if (!youtube) return raw
  url.searchParams.set("t", `${Math.max(0, Math.floor(seconds))}s`)
  return url.toString()
}

export function speakerHue(label: string): number {
  let value = 0
  for (const char of label) value = (value * 33 + char.charCodeAt(0)) % 360
  return value
}
