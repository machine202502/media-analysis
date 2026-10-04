import { BrowserRouter, Route, Routes } from "react-router-dom"
import { Library } from "./library"
import { Watch } from "./watch"

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Library />} />
        <Route path="/v/:id" element={<Watch />} />
      </Routes>
    </BrowserRouter>
  )
}
