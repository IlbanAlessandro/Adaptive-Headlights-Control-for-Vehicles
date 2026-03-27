"""
Script de Testare Model - Proiect 9
Testează modelul antrenat pe imagini și generează vizualizări

Autor: [Numele tău]
Data: Februarie 2025
"""

import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime


class ModelTester:
    """Clasă pentru testarea modelului YOLO antrenat."""

    def __init__(self, model_path='runs/detect/train/weights/best.pt',
                 test_images_folder='datasets/valid/images'):
        """
        Inițializare tester.

        Args:
            model_path: Calea către modelul antrenat
            test_images_folder: Folderul cu imagini de test
        """
        self.model_path = model_path
        self.test_images_folder = test_images_folder
        self.output_dir = 'test_results'
        os.makedirs(self.output_dir, exist_ok=True)

        # Încărcare model
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Model încărcat: {model_path}")
        else:
            print(f"❌ Modelul nu există la: {model_path}")
            self.model = None

    def test_on_random_images(self, num_images=5, confidence_threshold=0.25):
        """
        Testează modelul pe imagini aleatoare.

        Args:
            num_images: Numărul de imagini de testat
            confidence_threshold: Pragul minim de încredere pentru detecții
        """
        if not self.model:
            print("❌ Model neîncărcat!")
            return

        if not os.path.exists(self.test_images_folder):
            print(f"❌ Folderul {self.test_images_folder} nu există!")
            return

        # Găsire imagini
        all_images = [f for f in os.listdir(self.test_images_folder)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]

        if len(all_images) < num_images:
            num_images = len(all_images)
            print(f"⚠️  Doar {num_images} imagini disponibile")

        selected_images = random.sample(all_images, num_images)

        print(f"\n🔍 Testare pe {num_images} imagini aleatoare...")
        print(f"Confidence threshold: {confidence_threshold}")

        # Creare grid pentru vizualizare
        fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 6))

        if num_images == 1:
            axes = [axes]

        results_data = []

        for i, img_name in enumerate(selected_images):
            img_path = os.path.join(self.test_images_folder, img_name)

            # Predicție
            results = self.model(img_path, conf=confidence_threshold, verbose=False)

            # Desenare rezultate
            res_plotted = results[0].plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

            # Extragere informații
            boxes = results[0].boxes
            num_detections = len(boxes)

            detections_info = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                detections_info.append({
                    'class': cls_name,
                    'confidence': confidence
                })

            results_data.append({
                'image': img_name,
                'num_detections': num_detections,
                'detections': detections_info
            })

            # Afișare
            axes[i].imshow(res_rgb)
            axes[i].axis('off')
            title = f"{img_name}\nDetecții: {num_detections}"
            axes[i].set_title(title, fontsize=10, fontweight='bold')

        plt.suptitle('Rezultate Testare Model pe Imagini Aleatoare',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        # Salvare
        output_path = os.path.join(self.output_dir,
                                   f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Rezultate salvate: {output_path}")
        plt.close()

        # Afișare statistici
        self._print_test_statistics(results_data)

        return results_data

    def _print_test_statistics(self, results_data):
        """Afișează statistici despre testare."""
        print("\n" + "=" * 60)
        print("📊 STATISTICI TESTARE")
        print("=" * 60)

        total_detections = sum(r['num_detections'] for r in results_data)
        avg_detections = total_detections / len(results_data) if results_data else 0

        # Colectare clase detectate
        class_counts = {}
        confidences = []

        for result in results_data:
            for det in result['detections']:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
                confidences.append(det['confidence'])

        print(f"Total imagini testate: {len(results_data)}")
        print(f"Total detecții: {total_detections}")
        print(f"Medie detecții/imagine: {avg_detections:.2f}")

        if confidences:
            print(f"\nÎncredere medie: {np.mean(confidences):.3f}")
            print(f"Încredere min: {min(confidences):.3f}")
            print(f"Încredere max: {max(confidences):.3f}")

        if class_counts:
            print(f"\nDistribuție clase detectate:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {cls}: {count}")

        print("=" * 60 + "\n")

    def test_on_specific_image(self, image_path, confidence_threshold=0.25,
                               save_result=True):
        """
        Testează modelul pe o imagine specifică.

        Args:
            image_path: Calea către imagine
            confidence_threshold: Pragul de încredere
            save_result: Dacă True, salvează rezultatul
        """
        if not self.model:
            print("❌ Model neîncărcat!")
            return None

        if not os.path.exists(image_path):
            print(f"❌ Imaginea nu există: {image_path}")
            return None

        print(f"\n🔍 Testare pe: {os.path.basename(image_path)}")

        # Predicție
        results = self.model(image_path, conf=confidence_threshold, verbose=False)

        # Desenare și afișare
        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        # Extragere informații
        boxes = results[0].boxes
        num_detections = len(boxes)

        print(f"Detecții găsite: {num_detections}")

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  - {cls_name}: {confidence:.3f} @ [{x1},{y1},{x2},{y2}]")

        # Vizualizare
        plt.figure(figsize=(12, 8))
        plt.imshow(res_rgb)
        plt.axis('off')
        plt.title(f"Testare: {os.path.basename(image_path)} - {num_detections} detecții",
                  fontsize=14, fontweight='bold')

        if save_result:
            output_path = os.path.join(self.output_dir,
                                       f'test_{os.path.basename(image_path)}')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✅ Rezultat salvat: {output_path}")

        plt.close()

        return results

    def batch_test_and_compare(self, num_tests=3, images_per_test=5):
        """
        Efectuează mai multe teste și compară rezultatele.

        Args:
            num_tests: Numărul de teste de efectuat
            images_per_test: Imagini per test
        """
        print("\n" + "=" * 60)
        print("🔬 TESTARE ÎN LOT - COMPARAȚIE REZULTATE")
        print("=" * 60)

        all_results = []

        for test_num in range(num_tests):
            print(f"\n--- Test {test_num + 1}/{num_tests} ---")
            results = self.test_on_random_images(num_images=images_per_test)
            if results:
                all_results.append(results)

        # Analiză comparativă
        if all_results:
            self._compare_test_results(all_results)

    def _compare_test_results(self, all_results):
        """Compară rezultatele din mai multe teste."""
        print("\n" + "=" * 60)
        print("📊 COMPARAȚIE REZULTATE MULTIPLE TESTE")
        print("=" * 60)

        avg_detections_per_test = []
        avg_confidences_per_test = []

        for test_idx, test_results in enumerate(all_results):
            total_dets = sum(r['num_detections'] for r in test_results)
            avg_dets = total_dets / len(test_results)
            avg_detections_per_test.append(avg_dets)

            all_confs = []
            for result in test_results:
                all_confs.extend([d['confidence'] for d in result['detections']])

            if all_confs:
                avg_conf = np.mean(all_confs)
                avg_confidences_per_test.append(avg_conf)

        print(f"Medie detecții/imagine pe toate testele: {np.mean(avg_detections_per_test):.2f}")
        if avg_confidences_per_test:
            print(f"Medie încredere pe toate testele: {np.mean(avg_confidences_per_test):.3f}")

        print("=" * 60 + "\n")


def main():
    """Funcție principală pentru testare."""
    # Căutare model
    model_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if not model_path:
        print("❌ Nu am găsit modelul antrenat!")
        print("💡 Antrenează mai întâi modelul sau verifică calea.")
        return

    # Inițializare tester
    tester = ModelTester(
        model_path=model_path,
        test_images_folder='datasets/valid/images'
    )

    # Rulare teste
    print("\n" + "=" * 70)
    print(" " * 20 + "🧪 TESTARE MODEL 🧪")
    print("=" * 70)

    # Test 1: 5 imagini aleatoare
    tester.test_on_random_images(num_images=5, confidence_threshold=0.25)

    # Test 2: Batch testing (opțional - decomentează dacă vrei)
    # tester.batch_test_and_compare(num_tests=3, images_per_test=5)

    print("\n✅ Testare completă!")
    print(f"📁 Rezultatele sunt salvate în: {tester.output_dir}/")


if __name__ == "__main__":
    main()