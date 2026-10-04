import { useEffect, type ReactNode } from "react"

export function Modal({
  title,
  onClose,
  wide = false,
  children,
}: {
  title: string
  onClose: () => void
  wide?: boolean
  children: ReactNode
}) {
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <div
      className="modal-back"
      onPointerDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className={wide ? "modal wide" : "modal"} role="dialog" aria-modal="true" aria-label={title}>
        <h2>{title}</h2>
        {children}
      </div>
    </div>
  )
}
