import cv2
import numpy as np
import math
import os

# --- 1. CONFIGURARE VITEZĂ ȘI DISTANȚĂ ---

# De câte ori să fie mai încet? (1 = normal, 3 = de 3 ori mai lent)
FACTOR_INCETINIRE = 3

# La ce distanță (în pixeli) să schimbe faza?
DISTANTA_ACTIVARE = 350

video_source = 'video_test/test_video.mp4'
output_file = 'video_output/rezultat_slow_motion.mp4'

# --- 2. INIȚIALIZARE ---
os.makedirs('video_output', exist_ok=True)

cap = cv2.VideoCapture(video_source)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Luăm FPS-ul original (de obicei 30)
fps_original = int(cap.get(cv2.CAP_PROP_FPS))
if fps_original == 0: fps_original = 30

# Calculăm noul FPS pentru Slow Motion
# Dacă videoul are 30 fps și noi îl scriem cu 10 fps, va dura de 3 ori mai mult.
fps_slow = max(5, int(fps_original / FACTOR_INCETINIRE))

out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps_slow, (w, h))

print(f"🎥 Procesare Slow Motion ({FACTOR_INCETINIRE}x mai lent)...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. PROCESARE LOGICĂ (Aceeași ca înainte) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cars = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            x, y, cw, ch = cv2.boundingRect(c)
            center_x = x + cw // 2
            center_y = y + ch // 2
            cars.append((center_x, center_y))

            # Desenăm contur simplu
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (255, 255, 0), 2)

    # --- 4. CALCUL DISTANȚĂ ---
    beam_status = "HIGH BEAM"
    beam_color = (0, 255, 0)  # Verde
    dist_px = 0

    if len(cars) >= 2:
        c1 = cars[0]
        c2 = cars[1]

        # Distanța Euclidiană
        dist_px = math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

        # Linie între mașini
        cv2.line(frame, c1, c2, (255, 0, 255), 2)

        # Verificăm pragul
        if dist_px < DISTANTA_ACTIVARE:
            beam_status = "LOW BEAM (Detectat!)"
            beam_color = (0, 0, 255)  # Roșu

            # Text de avertizare sub linie
            mid_x = (c1[0] + c2[0]) // 2
            mid_y = (c1[1] + c2[1]) // 2
            cv2.putText(frame, f"Prea Aproape! ({int(dist_px)}px)", (mid_x - 60, mid_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- 5. DASHBOARD GRAFIC ---
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

    # Cerc indicator
    cv2.circle(frame, (60, 50), 30, beam_color, -1)

    # Text Stare Mare
    cv2.putText(frame, beam_status, (110, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, beam_color, 3)

    # Afișăm distanța curentă mic în colț
    if dist_px > 0:
        cv2.putText(frame, f"Distanta: {int(dist_px)}px", (w - 250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Scriem cadrul în fișier (va fi slow motion la redare)
    out.write(frame)
    cv2.imshow("Slow Motion Demo", frame)

    # Așteptăm mai mult între cadre pentru a încetini și vizualizarea live
    # 30ms e normal. 100ms e lent.
    timp_asteptare = int(1000 / fps_slow)

    if cv2.waitKey(timp_asteptare) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Gata! Videoclipul încetinit este salvat în: {output_file}")