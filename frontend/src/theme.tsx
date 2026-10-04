import { useState } from "react"

export type Theme = "dark" | "light"

const KEY = "va-theme"

function storedTheme(): Theme {
  const current = document.documentElement.dataset.theme
  if (current === "light" || current === "dark") return current
  return localStorage.getItem(KEY) === "dark" ? "dark" : "light"
}

function writeTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme
  localStorage.setItem(KEY, theme)
}

export function ThemeSwitch() {
  const [theme, setTheme] = useState<Theme>(storedTheme)

  function choose(next: Theme) {
    writeTheme(next)
    setTheme(next)
  }

  return (
    <div className="theme" role="group" aria-label="Тема">
      <button type="button" aria-pressed={theme === "light"} onClick={() => choose("light")}>
        Светлая
      </button>
      <button type="button" aria-pressed={theme === "dark"} onClick={() => choose("dark")}>
        Тёмная
      </button>
    </div>
  )
}
