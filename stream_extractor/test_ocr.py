"""
Script de teste: roda OCR numa imagem local para verificar extração.
Útil para calibrar antes de rodar na stream ao vivo.

Uso:
    python test_ocr.py <caminho_da_imagem>
"""

import sys
import json
import cv2
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent))

from ocr.extractor import OCRExtractor
from detection.detector import TableDetector
from config import settings


def test_image(image_path: str):
    print(f"\n{'='*55}")
    print(f" Testando OCR: {image_path}")
    print(f"{'='*55}\n")

    # Carrega imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Não foi possível abrir: {image_path}")
        return

    h, w = frame.shape[:2]
    print(f"📐 Resolução: {w}x{h}")

    # ── Detecção ──────────────────────────────────────────────
    print("\n[1] Testando detecção visual...")
    detector = TableDetector(settings.DETECTION, settings.ROI)
    detected, score = detector.is_stats_table_visible(frame)
    scores = detector.compute_detection_score(frame)

    print(f"  Score geral: {score:.3f} ({'✓ DETECTADO' if detected else '✗ NÃO detectado'})")
    print(f"  Breakdown:")
    for k, v in scores.items():
        if k != "final":
            bar = "█" * int(v * 20)
            print(f"    {k:20s}: {v:.3f} {bar}")

    # Tempo de jogo
    match_time = detector.extract_match_time(frame)
    print(f"\n  Tempo de jogo: {match_time or 'não detectado'}")

    # ── OCR ───────────────────────────────────────────────────
    print("\n[2] Rodando OCR...")
    extractor = OCRExtractor(settings.OCR, settings.ROI)
    data = extractor.extract_from_frame(frame)

    success = data.get("_extraction_success", False)
    print(f"\n  Status: {'✅ Sucesso' if success else '⚠️  Parcial / Falhou'}")
    print(f"  Engine: {data.get('ocr_engine', '?')}")
    print(f"  Confiança: {data.get('confidence_score', 0):.2f}")
    print(f"  Tempo: {data.get('extraction_duration_ms', 0)}ms")

    # ── Resultado estruturado ──────────────────────────────────
    print(f"\n[3] Dados extraídos:\n")

    team_home = data.get("team_home", "?")
    team_away = data.get("team_away", "?")
    goals_h = data.get("goals_home", "?")
    goals_a = data.get("goals_away", "?")
    match_t = data.get("match_time", "?")

    print(f"  MATCH: {team_home} {goals_h} x {goals_a} {team_away} | {match_t}")
    print(f"  XG TOTAL: {data.get('xg_total', '?')} | DIFF: {data.get('xg_diff', '?')}")
    print()

    stats = [
        ("Possession %",        "possession"),
        ("Ball Recovery Time",  "ball_recovery"),
        ("Shots",               "shots"),
        ("Expected Goals",      "expected_goals"),
        ("Passes",              "passes"),
        ("Tackles",             "tackles"),
        ("Tackles Won",         "tackles_won"),
        ("Interceptions",       "interceptions"),
        ("Saves",               "saves"),
        ("Fouls Committed",     "fouls_committed"),
        ("Offsides",            "offsides"),
        ("Corners",             "corners"),
        ("Free Kicks",          "free_kicks"),
        ("Penalty Kicks",       "penalty_kicks"),
        ("Yellow Cards",        "yellow_cards"),
    ]

    for label, key in stats:
        h_val = data.get(f"{key}_home", "-")
        a_val = data.get(f"{key}_away", "-")
        print(f"  {label:25s} | {str(h_val):>6} | {str(a_val):>6}")

    print()
    circles = [
        ("Dribble Success",  "dribble_success"),
        ("Shot Accuracy",    "shot_accuracy"),
        ("Pass Accuracy",    "pass_accuracy"),
    ]
    for label, key in circles:
        h_val = data.get(f"{key}_home", "-")
        a_val = data.get(f"{key}_away", "-")
        print(f"  {label:25s} | {str(h_val):>5}% | {str(a_val):>5}%")

    # Salva resultado em JSON
    output = {k: v for k, v in data.items() if not k.startswith("_") and k != "ocr_raw_text"}
    output_path = Path(image_path).stem + "_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Resultado salvo em: {output_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_ocr.py <imagem.jpg>")
        print("Exemplo: python test_ocr.py screenshot.jpg")
        sys.exit(1)

    test_image(sys.argv[1])
