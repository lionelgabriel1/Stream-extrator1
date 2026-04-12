"""
Ferramenta de calibração do ROI.
Use este script para ajustar as coordenadas da tabela a partir de um screenshot.

Uso:
    python calibrate.py <caminho_para_screenshot.jpg>

Controles:
    - Clique e arraste para desenhar ROI
    - R: Reset
    - S: Salva coordenadas
    - Q: Sair
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path

drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
current_roi_name = "stats_table"
saved_rois = {}

ROIS_TO_CALIBRATE = [
    ("score_header",     "Header com placar e tempo de jogo", (0, 0, 255)),
    ("stats_table",      "Tabela central de estatísticas",    (0, 255, 0)),
    ("circles_left",     "Círculos métricas ESQUERDA",        (255, 0, 0)),
    ("circles_right",    "Círculos métricas DIREITA",         (255, 165, 0)),
    ("full_stats_screen","Tela completa de stats",            (128, 0, 128)),
]


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y


def calibrate(image_path: str):
    global current_roi_name, saved_rois

    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"❌ Não foi possível abrir: {image_path}")
        sys.exit(1)

    h, w = img_original.shape[:2]
    print(f"\n📐 Imagem: {w}x{h}")
    print("=" * 50)

    for roi_idx, (roi_name, description, color) in enumerate(ROIS_TO_CALIBRATE):
        current_roi_name = roi_name
        print(f"\n[{roi_idx+1}/{len(ROIS_TO_CALIBRATE)}] Calibrando: {roi_name}")
        print(f"  → {description}")
        print("  Clique e arraste para selecionar a área.")
        print("  [S] Salvar  [R] Refazer  [Q] Pular")

        window_name = f"Calibração: {roi_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(w, 1280), min(h, 720))
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            display = img_original.copy()

            # Desenha ROIs já salvas
            for saved_name, roi in saved_rois.items():
                sx1 = int(roi["x"] * w)
                sy1 = int(roi["y"] * h)
                sx2 = int((roi["x"] + roi["w"]) * w)
                sy2 = int((roi["y"] + roi["h"]) * h)
                cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (100, 100, 100), 1)
                cv2.putText(display, saved_name, (sx1+2, sy1+12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Desenha ROI atual
            if fx > 0 and fy > 0:
                x1, y1 = min(ix, fx), min(iy, fy)
                x2, y2 = max(ix, fx), max(iy, fy)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                
                # Mostra coordenadas relativas
                rx = x1 / w
                ry = y1 / h
                rw = (x2 - x1) / w
                rh = (y2 - y1) / h
                info = f"{roi_name}: x={rx:.3f} y={ry:.3f} w={rw:.3f} h={rh:.3f}"
                cv2.putText(display, info, (10, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Label instrução
            cv2.putText(display, f"Selecionando: {description}",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, "[S] Salvar  [R] Refazer  [Q] Pular",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s') and fx > 0 and fy > 0:
                x1, y1 = min(ix, fx), min(iy, fy)
                x2, y2 = max(ix, fx), max(iy, fy)
                roi_data = {
                    "x": round(x1 / w, 4),
                    "y": round(y1 / h, 4),
                    "w": round((x2 - x1) / w, 4),
                    "h": round((y2 - y1) / h, 4),
                }
                saved_rois[roi_name] = roi_data
                print(f"  ✓ Salvo: {roi_data}")
                break

            elif key == ord('r'):
                print("  ↺ Refazendo...")
                # Reset coordenadas
                globals().update({"ix": -1, "iy": -1, "fx": -1, "fy": -1})

            elif key == ord('q'):
                print(f"  ⏭ Pulado: {roi_name}")
                break

        cv2.destroyWindow(window_name)

    # Salva resultado
    if saved_rois:
        output_path = "calibration_result.json"
        with open(output_path, "w") as f:
            json.dump(saved_rois, f, indent=4)

        print("\n" + "=" * 50)
        print(f"✅ Calibração salva em: {output_path}")
        print("\nCopie o seguinte para config/settings.py → ROI:\n")
        print("ROI = {")
        for name, roi in saved_rois.items():
            print(f'    "{name}": {{')
            print(f'        "x": {roi["x"]}, "y": {roi["y"]},')
            print(f'        "w": {roi["w"]}, "h": {roi["h"]},')
            print(f'    }},')
        print("}")
    else:
        print("\n⚠️  Nenhum ROI salvo.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python calibrate.py <screenshot.jpg>")
        print("Exemplo: python calibrate.py frames/capture_score0.85.jpg")
        sys.exit(1)

    calibrate(sys.argv[1])
