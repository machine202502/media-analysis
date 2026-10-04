export type VideoUsage = {
  total: number
  video: number
  files: number
  text: number
  vectors: number
  indexes: number
}

export type StageTime = {
  stage: string
  seconds: number
}

export type Video = {
  id: string
  title: string
  status: string
  error: string | null
  warning: string | null
  durationSec: number | null
  progress: number
  stageProgress: number
  startedAt: string | null
  processedSec: number | null
  createdAt: string
  queuePlace: number | null
  audio: boolean
  sourceUrl: string | null
  held: boolean
  diarize?: boolean
  merge?: boolean
  correct?: boolean
  stages: StageTime[]
  usage: VideoUsage
}

export type TimedWord = {
  text: string
  start: number
  end: number
}

export type Segment = {
  id: number
  position: number
  start: number
  end: number
  speaker: string | null
  speakerName: string
  text: string
  words: TimedWord[]
}

export type Speaker = {
  label: string
  name: string
}

export type Citation = {
  type?: string
  segmentId: number
  start: number
  end: number
  speakerName: string
  text: string
}

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
  citations: Citation[]
}

export type AgentHit = {
  start: number
  end: number
  speakerName: string
  text: string
}

export type AgentAction = {
  title: string
  detail: string
  hits?: AgentHit[]
  progress?: number
}

export type AgentMessage = {
  role: "user" | "assistant"
  content: string
  actions: AgentAction[]
  citations: Citation[]
}

export type SearchIndex = {
  id: string
  name: string
  kind: "moments" | "custom"
  instruction: string
  strict: boolean
  status: "building" | "ready" | "error"
  error: string | null
}

export type Detail = {
  video: Video
  speakers: Speaker[]
  segments: Segment[]
  chat: ChatMessage[]
  indexes: SearchIndex[]
  agent: AgentMessage[]
  chatPending?: boolean
  agentPending?: boolean
}

export type DialogState = {
  chat: ChatMessage[]
  agent: AgentMessage[]
  chatPending: boolean
  agentPending: boolean
}

export type Library = {
  videos: Video[]
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === "string") return body.detail
  } catch {
    /* ответ без json */
  }
  return "Запрос не выполнен"
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) throw new Error(await readError(response))
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export type Tune = {
  asrBatch: number
  diarBatch: number
  threads: number
  stageParallel: number
  memoryGb: number | null
  seenMemoryGb: number | null
  cores: number
}

export type Suggestion = {
  asrBatch: number
  diarBatch: number
  threads: number
  stageParallel: number
}

export function getSettings(): Promise<Tune> {
  return request("/api/settings")
}

export function saveSettings(body: {
  asrBatch: number
  diarBatch: number
  threads: number
  stageParallel: number
  memoryGb: number | null
}): Promise<Tune> {
  return request("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

export function suggestSettings(memoryGb: number): Promise<Suggestion> {
  return request("/api/settings/suggest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memoryGb }),
  })
}

export type StageChoice = {
  diarize: boolean
  merge: boolean
  correct: boolean
}

export function probeLink(url: string): Promise<{ durationSec: number | null; title: string }> {
  return request("/api/videos/probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  })
}

export function addByLink(url: string, stages: StageChoice): Promise<Video> {
  return request("/api/videos/link", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, ...stages }),
  })
}

export function listVideos(): Promise<Library> {
  return request("/api/videos")
}

export function getVideo(id: string): Promise<Detail> {
  return request(`/api/videos/${id}`)
}

export function removeVideo(id: string): Promise<void> {
  return request(`/api/videos/${id}`, { method: "DELETE" })
}

export function holdVideo(id: string, held: boolean): Promise<Video> {
  return request(`/api/videos/${id}/hold`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ held }),
  })
}

export function renameVideo(id: string, title: string): Promise<Video> {
  return request(`/api/videos/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  })
}

export function renameSpeaker(id: string, label: string, name: string): Promise<{ speakers: Speaker[] }> {
  return request(`/api/videos/${id}/speakers`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label, name }),
  })
}

export function clearChat(id: string): Promise<{ messages: ChatMessage[] }> {
  return request(`/api/videos/${id}/chat`, { method: "DELETE" })
}

export function askAgent(id: string, message: string, version = "3"): Promise<{ pending: boolean }> {
  return request(`/api/videos/${id}/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, version }),
  })
}

export function getDialog(id: string): Promise<DialogState> {
  return request(`/api/videos/${id}/dialog`)
}

export function stopDialog(id: string, kind: "chat" | "agent"): Promise<Pick<DialogState, "chatPending" | "agentPending">> {
  return request(`/api/videos/${id}/dialog/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind }),
  })
}

export function clearAgent(id: string): Promise<{ messages: AgentMessage[] }> {
  return request(`/api/videos/${id}/agent`, { method: "DELETE" })
}

export function askVideo(
  id: string,
  message: string,
  useLines: boolean,
  indexIds: string[],
): Promise<{ pending: boolean }> {
  return request(`/api/videos/${id}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, useLines, indexIds }),
  })
}

export function listIndexes(id: string): Promise<{ indexes: SearchIndex[] }> {
  return request(`/api/videos/${id}/indexes`)
}

export function buildMoments(id: string): Promise<SearchIndex> {
  return request(`/api/videos/${id}/indexes/moments`, { method: "POST" })
}

export function createIndex(
  id: string,
  name: string,
  instruction: string,
  strict = true,
): Promise<SearchIndex> {
  return request(`/api/videos/${id}/indexes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, instruction, strict }),
  })
}

export function rebuildIndex(id: string, indexId: string): Promise<SearchIndex> {
  return request(`/api/videos/${id}/indexes/${indexId}/rebuild`, { method: "POST" })
}

export function removeIndex(id: string, indexId: string): Promise<void> {
  return request(`/api/videos/${id}/indexes/${indexId}`, { method: "DELETE" })
}

export function uploadVideos(
  files: File[],
  stages: StageChoice,
  onProgress: (ratio: number) => void,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const body = new FormData()
    for (const file of files) body.append("files", file)
    body.append("diarize", String(stages.diarize))
    body.append("merge", String(stages.merge))
    body.append("correct", String(stages.correct))
    const xhr = new XMLHttpRequest()
    xhr.open("POST", "/api/videos")
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total)
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve()
        return
      }
      let detail = "Не удалось загрузить"
      try {
        const parsed = JSON.parse(xhr.responseText) as { detail?: unknown }
        if (typeof parsed.detail === "string") detail = parsed.detail
      } catch {
        /* не json */
      }
      reject(new Error(detail))
    }
    xhr.onerror = () => reject(new Error("Сеть недоступна"))
    xhr.send(body)
  })
}
