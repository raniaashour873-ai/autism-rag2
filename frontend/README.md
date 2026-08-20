# Frontend

The **live demo UI** is `index.html` (vanilla HTML/JS). No npm build is required.

Local: run FastAPI on port 8000, then open `index.html` or visit `http://localhost:8000/ui`.

Production: serve `/ui` from the same FastAPI origin so the UI calls `/ask` and `/routine` without a hardcoded localhost URL. Set `FRONTEND_URL` to that public origin.

`src/` is an unused React/Vite prototype. Do not treat `npm run dev` as the product UI.
