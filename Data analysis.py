"""
Script de Analiză Completă a Datelor - Proiect 9
Generează toate statisticile și graficele pentru documentație

Autor: [Numele tău]
Data: Februarie 2025
"""

import os
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class DatasetAnalyzer:
    """Analizor complet pentru setul de date și rezultatele antrenamentului."""

    def __init__(self, dataset_path='datasets', runs_path='runs/detect'):
        """
        Inițializare analizor.

        Args:
            dataset_path: Calea către folderul cu date
            runs_path: Calea către folderul cu rezultate antrenament
        """
        self.dataset_path = dataset_path
        self.runs_path = runs_path
        self.output_dir = 'analysis_output'
        os.makedirs(self.output_dir, exist_ok=True)

        # Statistici dataset
        self.train_path = os.path.join(dataset_path, 'train', 'images')
        self.valid_path = os.path.join(dataset_path, 'valid', 'images')

    def analyze_dataset_distribution(self):
        """Analizează și vizualizează distribuția setului de date."""
        print("\n" + "=" * 60)
        print("📊 ANALIZA SETULUI DE DATE")
        print("=" * 60)

        if not os.path.exists(self.train_path) or not os.path.exists(self.valid_path):
            print("❌ Folderele de date nu există!")
            return None

        # Numărare imagini
        num_train = len([f for f in os.listdir(self.train_path)
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
        num_valid = len([f for f in os.listdir(self.valid_path)
                         if f.endswith(('.jpg', '.png', '.jpeg'))])

        total = num_train + num_valid

        print(f"Imagini de antrenament: {num_train}")
        print(f"Imagini de validare: {num_valid}")
        print(f"Total imagini: {total}")
        print(f"Ratio antrenament/validare: {num_train / total * 100:.1f}% / {num_valid / total * 100:.1f}%")

        # Creare grafic
        labels = ['Antrenament', 'Validare']
        sizes = [num_train, num_valid]
        colors = ['#66b3ff', '#99ff99']
        explode = (0.05, 0.05)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie Chart
        ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title('Distribuția Datelor în Set', fontsize=14, fontweight='bold')

        # Bar Chart
        bars = ax2.bar(labels, sizes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.set_ylabel('Număr de Imagini', fontsize=12)
        ax2.set_title('Comparație Cantitativă', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        # Adăugare valori pe bare
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'dataset_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafic salvat: {output_path}")
        plt.close()

        return {'train': num_train, 'valid': num_valid, 'total': total}

    def show_sample_images(self, num_samples=6):
        """Afișează imagini de exemplu din dataset."""
        print("\n" + "=" * 60)
        print("🖼️  IMAGINI DE EXEMPLU DIN DATASET")
        print("=" * 60)

        if not os.path.exists(self.train_path):
            print("❌ Folderul de antrenament nu există!")
            return

        # Selectare imagini aleatoare
        all_images = [f for f in os.listdir(self.train_path)
                      if f.endswith(('.jpg', '.png', '.jpeg'))]

        if len(all_images) < num_samples:
            num_samples = len(all_images)

        selected = random.sample(all_images, num_samples)

        # Creare grid
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, img_name in enumerate(selected):
            img_path = os.path.join(self.train_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axes[idx].imshow(img)
            axes[idx].set_title(f"{img_name}\n{img.shape[1]}x{img.shape[0]}",
                                fontsize=9)
            axes[idx].axis('off')

        plt.suptitle('Exemple de Imagini din Setul de Antrenament',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'sample_images.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Imagini salvate: {output_path}")
        plt.close()

    def analyze_image_sizes(self):
        """Analizează dimensiunile imaginilor din dataset."""
        print("\n" + "=" * 60)
        print("📐 ANALIZA DIMENSIUNILOR IMAGINILOR")
        print("=" * 60)

        widths = []
        heights = []
        aspects = []

        for img_name in os.listdir(self.train_path):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(self.train_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)

        if not widths:
            print("❌ Nu am găsit imagini valide!")
            return

        print(f"Lățime medie: {np.mean(widths):.0f} px (min: {min(widths)}, max: {max(widths)})")
        print(f"Înălțime medie: {np.mean(heights):.0f} px (min: {min(heights)}, max: {max(heights)})")
        print(f"Aspect ratio mediu: {np.mean(aspects):.2f}")

        # Creare histograme
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].hist(widths, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(widths), color='red', linestyle='--', linewidth=2,
                        label=f'Medie: {np.mean(widths):.0f}')
        axes[0].set_xlabel('Lățime (px)', fontsize=11)
        axes[0].set_ylabel('Frecvență', fontsize=11)
        axes[0].set_title('Distribuția Lățimilor', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3, linestyle='--')

        axes[1].hist(heights, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.mean(heights), color='red', linestyle='--', linewidth=2,
                        label=f'Medie: {np.mean(heights):.0f}')
        axes[1].set_xlabel('Înălțime (px)', fontsize=11)
        axes[1].set_ylabel('Frecvență', fontsize=11)
        axes[1].set_title('Distribuția Înălțimilor', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle='--')

        axes[2].hist(aspects, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.mean(aspects), color='red', linestyle='--', linewidth=2,
                        label=f'Medie: {np.mean(aspects):.2f}')
        axes[2].set_xlabel('Aspect Ratio (W/H)', fontsize=11)
        axes[2].set_ylabel('Frecvență', fontsize=11)
        axes[2].set_title('Distribuția Aspect Ratio', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'image_dimensions.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grafic salvat: {output_path}")
        plt.close()

    def find_latest_training_run(self):
        """Găsește cel mai recent folder de antrenament."""
        if not os.path.exists(self.runs_path):
            return None

        dirs = [d for d in os.listdir(self.runs_path) if 'train' in d]
        if not dirs:
            return None

        latest = sorted(dirs, key=lambda x: os.path.getmtime(
            os.path.join(self.runs_path, x)))[-1]

        return os.path.join(self.runs_path, latest)

    def display_training_results(self):
        """Afișează rezultatele antrenamentului."""
        print("\n" + "=" * 60)
        print("📈 REZULTATE ANTRENAMENT MODEL")
        print("=" * 60)

        latest_run = self.find_latest_training_run()

        if not latest_run:
            print("❌ Nu am găsit rezultate de antrenament!")
            return

        print(f"Folder antrenament: {os.path.basename(latest_run)}")

        # Paths către grafice
        results_img = os.path.join(latest_run, 'results.png')
        confusion_matrix = os.path.join(latest_run, 'confusion_matrix.png')

        # Copiere grafice în folderul de analiză
        if os.path.exists(results_img):
            output_path = os.path.join(self.output_dir, 'training_results.png')
            img = cv2.imread(results_img)
            cv2.imwrite(output_path, img)
            print(f"✅ Rezultate antrenament copiate: {output_path}")

        if os.path.exists(confusion_matrix):
            output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            img = cv2.imread(confusion_matrix)
            cv2.imwrite(output_path, img)
            print(f"✅ Matrice confuzie copiată: {output_path}")

    def generate_full_report(self):
        """Generează raport complet cu toate analizele."""
        print("\n" + "=" * 70)
        print(" " * 15 + "🚗 RAPORT COMPLET ANALIZA DATE 🚗")
        print("=" * 70)
        print(f"Data generare: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Proiect: Control Adaptiv Faruri (Proiect 9)")
        print("=" * 70)

        # Rulare toate analizele
        dataset_stats = self.analyze_dataset_distribution()
        self.show_sample_images(num_samples=6)
        self.analyze_image_sizes()
        self.display_training_results()

        # Salvare statistici în JSON
        stats = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_stats,
            'output_directory': self.output_dir
        }

        stats_file = os.path.join(self.output_dir, 'dataset_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "=" * 70)
        print("✅ RAPORT COMPLET GENERAT!")
        print(f"📁 Toate fișierele sunt în: {self.output_dir}/")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    # Inițializare și generare raport
    analyzer = DatasetAnalyzer(
        dataset_path='datasets',
        runs_path='runs/detect'
    )

    analyzer.generate_full_report()