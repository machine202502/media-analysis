import type { Speaker } from "./api"

export function mentionAt(value: string, cursor: number): { start: number; query: string } | null {
  const before = value.slice(0, cursor)
  const match = /(^|\s)@([^\s@]*)$/.exec(before)
  if (!match) return null
  return { start: before.length - match[2].length - 1, query: match[2] }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export function labelsToMentions(text: string, speakers: Speaker[]): string {
  const ordered = [...speakers].sort((left, right) => right.label.length - left.label.length)
  let result = text
  for (const speaker of ordered) {
    const name = speaker.name.trim()
    if (!speaker.label || !name) continue
    const pattern = new RegExp(`(?<!@)\\b${escapeRegExp(speaker.label)}\\b`, "g")
    result = result.replace(pattern, `@${name}`)
  }
  return result
}

export function mentionPattern(speakers: Speaker[]): RegExp | null {
  const names = [...new Set(speakers.map((speaker) => speaker.name.trim()).filter(Boolean))].sort(
    (left, right) => right.length - left.length,
  )
  if (names.length === 0) return null
  return new RegExp(`@(?:${names.map(escapeRegExp).join("|")})`, "gi")
}
