import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react"

const KEY = "va-panes"
type Pane = "dialog" | "video" | "chat"

const DEFAULTS: Record<Pane, number> = { dialog: 1.15, video: 1, chat: 0.95 }

function readSizes(): Record<Pane, number> {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) || "") as Partial<Record<Pane, number>>
    if ((["dialog", "video", "chat"] as const).every((pane) => typeof parsed[pane] === "number" && parsed[pane]! > 0)) {
      return parsed as Record<Pane, number>
    }
  } catch {
    /* оставляем доли по умолчанию */
  }
  return { ...DEFAULTS }
}

type Sash = { order: number; left: Pane; right: Pane }

function placement(showVideo: boolean, narrow: boolean) {
  if (!showVideo) {
    return {
      dialog: 1,
      video: 0,
      chat: 3,
      first: { order: 2, left: "dialog", right: "chat" } satisfies Sash,
      second: null as Sash | null,
    }
  }
  if (narrow) {
    return {
      video: 1,
      dialog: 3,
      chat: 5,
      first: { order: 2, left: "video", right: "dialog" } satisfies Sash,
      second: { order: 4, left: "dialog", right: "chat" } satisfies Sash,
    }
  }
  return {
    dialog: 1,
    video: 3,
    chat: 5,
    first: { order: 2, left: "dialog", right: "video" } satisfies Sash,
    second: { order: 4, left: "video", right: "chat" } satisfies Sash,
  }
}

export function Panes({
  dialog,
  video,
  chat,
  showVideo,
  cinema,
}: {
  dialog: ReactNode
  video: ReactNode
  chat: ReactNode
  showVideo: boolean
  cinema: boolean
}) {
  const stageRef = useRef<HTMLDivElement>(null)
  const sizes = useRef(readSizes())
  const [, redraw] = useState(0)
  const [narrow, setNarrow] = useState(() => window.matchMedia("(max-width: 980px)").matches)

  useEffect(() => {
    const media = window.matchMedia("(max-width: 980px)")
    const apply = () => setNarrow(media.matches)
    media.addEventListener("change", apply)
    return () => media.removeEventListener("change", apply)
  }, [])

  const place = placement(showVideo, narrow)

  function drag(pair: Sash, event: ReactPointerEvent<HTMLDivElement>) {
    const stage = stageRef.current
    if (!stage) return
    event.preventDefault()
    const sash = event.currentTarget
    const vertical = stage.classList.contains("narrow")
    const start = vertical ? event.clientY : event.clientX
    const sashCount = showVideo ? 2 : 1
    const span = Math.max(1, (vertical ? stage.clientHeight : stage.clientWidth) - sashCount * 8)
    const origin = { ...sizes.current }
    const visible: Pane[] = showVideo ? ["dialog", "video", "chat"] : ["dialog", "chat"]
    const total = visible.reduce((sum, pane) => sum + origin[pane], 0)
    const min = (150 / span) * total
    sash.setPointerCapture(event.pointerId)
    sash.classList.add("active")
    document.body.classList.add(vertical ? "resizing-row" : "resizing-col")

    function move(ev: PointerEvent) {
      const point = vertical ? ev.clientY : ev.clientX
      const delta = ((point - start) / span) * total
      let leading = origin[pair.left] + delta
      let trailing = origin[pair.right] - delta
      if (leading < min) {
        trailing -= min - leading
        leading = min
      }
      if (trailing < min) {
        leading -= min - trailing
        trailing = min
      }
      sizes.current = { ...sizes.current, [pair.left]: leading, [pair.right]: trailing }
      redraw((value) => value + 1)
    }

    function stop() {
      sash.classList.remove("active")
      document.body.classList.remove("resizing-row", "resizing-col")
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", stop)
      localStorage.setItem(KEY, JSON.stringify(sizes.current))
    }

    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", stop)
  }

  function sash(pair: Sash | null) {
    if (!pair) return <div className="sash off" />
    return (
      <div
        className="sash"
        style={{ order: pair.order }}
        role="separator"
        aria-orientation={narrow ? "horizontal" : "vertical"}
        aria-label="Изменить размер панелей"
        onPointerDown={(event) => drag(pair, event)}
      />
    )
  }

  const stageClass = `${narrow ? "stage narrow" : "stage"}${cinema ? " cinema" : ""}`
  return (
    <div className={stageClass} ref={stageRef}>
      {!cinema && (
        <div className="pane" style={{ flexGrow: sizes.current.dialog, order: place.dialog }}>
          {dialog}
        </div>
      )}
      {!cinema && sash(place.first)}
      <div
        className={showVideo ? "pane" : "audio-only"}
        style={showVideo ? { flexGrow: cinema ? 1 : sizes.current.video, order: cinema ? 1 : place.video } : undefined}
      >
        {video}
      </div>
      {!cinema && sash(place.second)}
      {!cinema && (
        <div className="pane" style={{ flexGrow: sizes.current.chat, order: place.chat }}>
          {chat}
        </div>
      )}
    </div>
  )
}
