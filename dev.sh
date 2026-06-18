#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/.venv/bin/uvicorn" ]]; then
  echo "Brak środowiska Python. Utwórz je: python -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Brak zależności frontendu. Uruchom: cd frontend && npm install"
  exit 1
fi

# Vite 8 wymaga nowszego Node; przełączamy na major 20 przez nvm (jeśli dostępne).
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "$NVM_DIR/nvm.sh" ]]; then
  # shellcheck source=/dev/null
  source "$NVM_DIR/nvm.sh"
  nvm use 20
else
  echo "Uwaga: nie znaleziono nvm ($NVM_DIR/nvm.sh). Używam bieżącego Node: $(node -v 2>/dev/null || echo unknown)"
fi

cleanup() {
  trap - INT TERM EXIT
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Backend:  http://127.0.0.1:8000"
echo "Frontend: http://127.0.0.1:5173"
echo "Zatrzymaj oba serwisy: Ctrl+C"
echo

(
  "$ROOT/.venv/bin/uvicorn" backend.app.main:app --reload --host 127.0.0.1 --port 8000 2>&1 \
    | sed -u 's/^/[backend] /'
) &

(
  cd "$ROOT/frontend"
  npm run dev 2>&1 | sed -u 's/^/[frontend] /'
) &

wait
