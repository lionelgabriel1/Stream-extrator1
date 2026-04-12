# Stream Stats Extractor

Sistema automatizado de extração de estatísticas de partidas de eFootball/FIFA
transmitidas em streams da Twitch e YouTube.

## Como funciona

```
Stream (Twitch/YouTube)
        │
        ▼
   streamlink         ← obtém URL direta do vídeo
        │
        ▼
     ffmpeg           ← decodifica frames via pipe (sem navegador)
        │
        ▼
  Frame Queue         ← fila thread-safe de frames
        │
        ▼
  Detector (OpenCV)   ← analisa frames em baixa frequência
        │
   Estado: IDLE (0.5 FPS)
        │ tempo > 85min
        ▼
   Estado: ALERT (5 FPS)
        │ tabela detectada 2x consecutivas
        ▼
   Estado: CAPTURE
        │ burst 15 FPS por 3 segundos
        ▼
  Melhor frame selecionado
        │
        ▼
  OCR (Tesseract)     ← só roda aqui, não continuamente
        │ fallback
  OCR (EasyOCR)
        │
        ▼
  Parser              ← estrutura os dados
        │
        ▼
  PostgreSQL/SQLite   ← salva com timestamp
        │
        ▼
  Cooldown 120s       ← aguarda próxima partida
        │
        └──────────────────────────────────────────► volta para IDLE
```

## Estrutura de pastas

```
stream_extractor/
├── main.py                  ← Entry point, orquestrador multiprocessing
├── calibrate.py             ← Ferramenta de calibração de ROI (GUI)
├── test_ocr.py              ← Testa OCR num screenshot local
├── setup.sh                 ← Instala todas as dependências
├── requirements.txt
│
├── config/
│   └── settings.py          ← ⚙️  CONFIGURAÇÃO PRINCIPAL (edite aqui)
│
├── core/
│   └── capture.py           ← Captura de frames via streamlink + ffmpeg
│
├── detection/
│   └── detector.py          ← Detecção da tabela via OpenCV
│
├── ocr/
│   └── extractor.py         ← OCR + parser de estatísticas
│
├── database/
│   └── manager.py           ← PostgreSQL / SQLite
│
├── workers/
│   └── stream_worker.py     ← Loop principal por stream
│
├── utils/
│   └── logger.py            ← Logging com rotação
│
├── logs/                    ← Logs automáticos (criado na execução)
├── frames/                  ← Frames capturados (criado na execução)
└── data/                    ← SQLite DB se aplicável
```

## Instalação

### Pré-requisitos do sistema
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg tesseract-ocr tesseract-ocr-eng

# macOS
brew install ffmpeg tesseract
```

### Setup completo
```bash
chmod +x setup.sh
./setup.sh
```

### Ou manualmente
```bash
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Configuração

Edite `config/settings.py`:

```python
# 1. Stream
STREAMS = [
    {
        "id": "esbfootball",
        "url": "https://www.twitch.tv/esbfootball",
        "quality": "720p",
        "enabled": True,
    }
]

# 2. Banco de dados
DATABASE = {
    "type": "postgresql",    # ou "sqlite"
    "host": "localhost",
    "name": "stream_stats",
    "user": "postgres",
    "password": "sua_senha",
}
```

## Execução

```bash
source venv/bin/activate

# Rodar o sistema principal
python main.py

# Testar OCR num screenshot antes de rodar ao vivo
python test_ocr.py screenshot.jpg

# Calibrar coordenadas ROI visualmente
python calibrate.py screenshot.jpg
```

## Banco de dados

### Tabela principal: `matches`

| Campo | Tipo | Descrição |
|-------|------|-----------|
| id | INTEGER | PK autoincrement |
| stream_id | VARCHAR | ID da stream |
| team_home / team_away | VARCHAR | Nomes dos times |
| player_home / player_away | VARCHAR | Nomes dos jogadores |
| goals_home / goals_away | INTEGER | Placar |
| match_time | VARCHAR | Tempo (ex: "91:43") |
| possession_home/away | DECIMAL | Posse de bola % |
| shots_home/away | DECIMAL | Chutes |
| expected_goals_home/away | DECIMAL | xG |
| passes_home/away | DECIMAL | Passes |
| tackles_home/away | DECIMAL | Duelos |
| ... | ... | (todos os stats) |
| dribble_success_home/away | DECIMAL | % sucesso drible |
| shot_accuracy_home/away | DECIMAL | % precisão chute |
| pass_accuracy_home/away | DECIMAL | % precisão passe |
| xg_total | DECIMAL | xG total da partida |
| xg_diff | DECIMAL | Diferença de xG |
| captured_at | TIMESTAMP | Data/hora da captura |
| frame_path | VARCHAR | Caminho do frame salvo |
| ocr_engine | VARCHAR | "tesseract" ou "easyocr" |
| confidence_score | DECIMAL | Confiança do OCR |

### Consultas úteis

```sql
-- Últimas 10 partidas
SELECT team_home, goals_home, goals_away, team_away, captured_at
FROM matches ORDER BY captured_at DESC LIMIT 10;

-- Partidas com maior xG total
SELECT team_home, team_away, xg_total, goals_home, goals_away
FROM matches ORDER BY xg_total DESC LIMIT 20;

-- Estatísticas por stream
SELECT stream_id, COUNT(*) as total, MAX(captured_at) as ultima
FROM matches GROUP BY stream_id;
```

## Calibração de ROI

Se o layout da sua stream for diferente do padrão ESB Football:

```bash
# 1. Tire um screenshot da tabela (pode usar test_ocr.py primeiro)
# 2. Rode o calibrador visual
python calibrate.py screenshot.jpg

# 3. Selecione cada região com o mouse
# 4. Copie as coordenadas geradas para config/settings.py → ROI
```

## Ajuste de parâmetros

Em `config/settings.py → DETECTION`:

```python
DETECTION = {
    "fps_idle": 0.5,         # ↑ mais frames = mais CPU
    "fps_alert": 5.0,        # FPS quando jogo > 85min
    "fps_capture": 15.0,     # FPS durante burst
    "alert_minute": 85,      # ↓ ativa alerta mais cedo
    "detection_threshold": 0.65,  # ↓ mais sensível (mais falsos positivos)
    "burst_window_seconds": 3.0,  # ↑ mais frames para escolher
    "post_capture_cooldown_seconds": 120,  # Espera entre partidas
}
```

## Troubleshooting

**streamlink não encontra a stream**
```bash
streamlink --stream-url https://www.twitch.tv/esbfootball best
```

**Tesseract não instalado**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng
# Verificar: tesseract --version
```

**OCR com baixa precisão**
- Use `python test_ocr.py screenshot.jpg` para debugar
- Ajuste `preprocessing.scale_factor` e `contrast_enhance` em settings.py
- Calibre o ROI com `python calibrate.py screenshot.jpg`

**Tabela não detectada**
- Reduza `detection_threshold` de 0.65 para 0.50
- Verifique se o ROI `stats_table` está correto

**Duplicatas no banco**
- Aumente `post_capture_cooldown_seconds`
