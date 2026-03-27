import cv2
import os
from ultralytics import YOLO

# --- CONFIGURARE ---
# Calea către modelul tău antrenat (verifică să fie cel corect)
model_path = 'runs/detect/train_finetune6/weights/best.pt'

# Folderul cu pozele REALE din setul de date (nu cele din simulare)
# Asigură-te că aici sunt poze cu mașini reale
folder_poze = 'datasets_video/test/images'

# Unde salvăm rezultatele
output_folder = 'rezultate_documentatie'
os.makedirs(output_folder, exist_ok=True)

# Clasele care declanșează faza scurtă
target_classes = ['headlight', 'car', 'truck', 'bus', 'taillight']

print(f"🧠 Încarc modelul: {model_path}")
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"❌ Eroare: Nu găsesc modelul! {e}")
    exit()

# Luăm pozele
all_files = os.listdir(folder_poze)
images_to_process = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg'))]

print(f"📸 Am găsit {len(images_to_process)} imagini. Încep procesarea...")

for i, img_name in enumerate(images_to_process):
    img_path = os.path.join(folder_poze, img_name)
    img = cv2.imread(img_path)

    if img is None: continue

    # Facem detecția
    results = model(img, verbose=False, conf=0.25)
    traffic_detected = False

    # Desenăm cutiile
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])  # Scorul de încredere (ex: 0.85)

            if cls_name in target_classes:
                traffic_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Culoare: Albastru pentru mașini (ca la coleg), Roșu pentru faruri
                color_box = (255, 0, 0) if cls_name == 'car' else (0, 255, 255)

                # 1. Desenăm Pătratul
                cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)

                # 2. Scriem Textul cu Scorul (ex: "car 0.88") - EXACT CA IN PDF
                label = f"{cls_name} {conf:.2f}"

                # Facem un fundal mic la text ca să se citească
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(img, (x1, y1 - 20), (x1 + t_size[0], y1), color_box, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- DASHBOARD JOS (Decizia Sistemului) ---
    h, w, _ = img.shape
    # Bandă neagră jos
    cv2.rectangle(img, (0, h - 80), (w, h), (0, 0, 0), -1)

    if traffic_detected:
        text_status = "DETECTAT: Vehicul Identificat"
        text_action = ">> FAZA SCURTA <<"
        color_status = (0, 0, 255)  # Roșu
    else:
        text_status = "Drum Liber"
        text_action = ">> FAZA LUNGA <<"
        color_status = (0, 255, 0)  # Verde

    # Scriem stânga sus pe bandă
    cv2.putText(img, text_status, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    # Scriem mare acțiunea
    cv2.putText(img, text_action, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_status, 2)

    # Salvăm
    save_path = os.path.join(output_folder, f"analizat_{img_name}")
    cv2.imwrite(save_path, img)

    if i % 10 == 0:
        print(f"✅ Procesat {i}/{len(images_to_process)}...")

print(f"🎉 Gata! Verifică folderul '{output_folder}' și alege cele mai bune poze pentru Word.")