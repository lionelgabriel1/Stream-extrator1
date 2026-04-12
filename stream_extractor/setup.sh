#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  setup.sh - Instala dependências do sistema e Python
# ═══════════════════════════════════════════════════════════

set -e

echo "════════════════════════════════════════"
echo " Stream Stats Extractor - Setup"
echo "════════════════════════════════════════"

# ── 1. Dependências do sistema ──────────────────────────────
echo "[1/4] Instalando dependências do sistema..."
sudo apt-get update -q
sudo apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0

echo "  ✓ ffmpeg, tesseract instalados"

# ── 2. Ambiente virtual Python ──────────────────────────────
echo "[2/4] Criando ambiente virtual Python..."
python3 -m venv venv
source venv/bin/activate
echo "  ✓ venv criado"

# ── 3. Dependências Python ──────────────────────────────────
echo "[3/4] Instalando dependências Python..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Dependências Python instaladas"

# ── 4. Cria diretórios necessários ──────────────────────────
echo "[4/4] Criando diretórios..."
mkdir -p logs frames data
echo "  ✓ Diretórios criados: logs/, frames/, data/"

echo ""
echo "════════════════════════════════════════"
echo " ✅ Setup concluído!"
echo ""
echo " Próximos passos:"
echo "   1. Edite config/settings.py"
echo "      - Configure DATABASE (host, user, password)"
echo "      - Confirme a URL da stream em STREAMS"
echo ""
echo "   2. Para PostgreSQL, crie o banco:"
echo "      createdb stream_stats"
echo ""
echo "   3. Execute:"
echo "      source venv/bin/activate"
echo "      python main.py"
echo "════════════════════════════════════════"
