#!/usr/bin/env bash
# Compile main.tex to PDF.
# Prefer tectonic (DreamORM style), with local and Docker fallbacks.

set -euo pipefail

TEX_FILE="main.tex"
TEX_BASENAME="${TEX_FILE%.tex}"
BUILD_DIR="build"
PDF_OUT="${BUILD_DIR}/${TEX_BASENAME}.pdf"

mkdir -p "${BUILD_DIR}"

run_pdflatex() {
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="${BUILD_DIR}" "${TEX_FILE}" >/dev/null
}

if command -v tectonic >/dev/null 2>&1; then
  echo "[compile] Using tectonic"
  tectonic -X compile --outdir "${BUILD_DIR}" "${TEX_FILE}"
elif command -v latexmk >/dev/null 2>&1; then
  echo "[compile] Using latexmk"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="${BUILD_DIR}" "${TEX_FILE}"
elif command -v pdflatex >/dev/null 2>&1; then
  echo "[compile] Using pdflatex fallback"
  run_pdflatex
  if [[ -f "${TEX_BASENAME}.bib" ]]; then
    (cd "${BUILD_DIR}" && bibtex "${TEX_BASENAME}" >/dev/null)
  fi
  run_pdflatex
  run_pdflatex
elif command -v docker >/dev/null 2>&1; then
  echo "[compile] Using Docker fallback (texlive/texlive:latest)"
  docker run --rm -v "$(pwd)":/work -w /work texlive/texlive:latest \
    sh -lc "pdflatex -interaction=nonstopmode -halt-on-error -output-directory='${BUILD_DIR}' '${TEX_FILE}' && pdflatex -interaction=nonstopmode -halt-on-error -output-directory='${BUILD_DIR}' '${TEX_FILE}'"
else
  echo "Error: No LaTeX compiler found (tectonic/latexmk/pdflatex/docker)." >&2
  exit 1
fi

echo ""
echo "Build complete. Artifacts are in '${BUILD_DIR}'."
echo "PDF location: ${PDF_OUT}"
