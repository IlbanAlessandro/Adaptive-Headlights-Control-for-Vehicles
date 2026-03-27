"""
Script de Setup și Verificare - Proiect 9
Verifică toate dependințele și structura proiectului

Autor: [Numele tău]
Data: Februarie 2025
"""

import os
import sys
import importlib



class ProjectSetup:
    """Verificare și setup proiect."""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def check_python_version(self):
        """Verifică versiunea Python."""
        print("\n🐍 Verificare Python...")
        version = sys.version_info

        if version.major == 3 and version.minor >= 8:
            self.info.append(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.errors.append(f"❌ Python {version.major}.{version.minor} - Necesită 3.8+")

    def check_dependencies(self):
        """Verifică dacă toate bibliotecile sunt instalate."""
        print("\n📦 Verificare Dependințe...")

        required_packages = {
            'cv2': 'opencv-python',
            'ultralytics': 'ultralytics',
            'torch': 'torch',
            'numpy': 'numpy',
            'matplotlib': 'matplotlib',
            'pandas': 'pandas'
        }

        for module, package in required_packages.items():
            try:
                importlib.import_module(module)
                self.info.append(f"✅ {package}")
            except ImportError:
                self.errors.append(f"❌ {package} - Rulează: pip install {package}")

    def check_directory_structure(self):
        """Verifică structura de foldere."""
        print("\n📁 Verificare Structură Foldere...")

        required_dirs = [
            'datasets',
            'datasets/train',
            'datasets/train/images',
            'datasets/valid',
            'datasets/valid/images',
            'video_test'
        ]

        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                # Numără fișiere
                if 'images' in dir_path:
                    num_files = len([f for f in os.listdir(dir_path)
                                     if f.endswith(('.jpg', '.png', '.jpeg'))])
                    self.info.append(f"✅ {dir_path}/ ({num_files} imagini)")
                else:
                    self.info.append(f"✅ {dir_path}/")
            else:
                self.warnings.append(f"⚠️  {dir_path}/ - Nu există")

    def check_video_files(self):
        """Verifică existența fișierelor video."""
        print("\n🎥 Verificare Fișiere Video...")

        video_folder = 'video_test'

        if not os.path.exists(video_folder):
            self.warnings.append(f"⚠️  Folder {video_folder}/ lipsește")
            return

        videos = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

        if videos:
            for video in videos:
                self.info.append(f"✅ {video}")
        else:
            self.warnings.append(f"⚠️  Nu există fișiere .mp4 în {video_folder}/")

    def check_trained_model(self):
        """Verifică dacă există model antrenat."""
        print("\n🤖 Verificare Model Antrenat...")

        model_paths = [
            'runs/detect/train/weights/best.pt',
            'runs/detect/train2/weights/best.pt',
            'runs/detect/train3/weights/best.pt'
        ]

        found = False
        for path in model_paths:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self.info.append(f"✅ Model găsit: {path} ({size_mb:.1f} MB)")
                found = True
                break

        if not found:
            self.warnings.append("⚠️  Nu există model antrenat - Antrenează modelul mai întâi")

    def check_data_yaml(self):
        """Verifică fișierul de configurare YOLO."""
        print("\n⚙️  Verificare Configurare Dataset...")

        yaml_path = 'datasets/data.yaml'

        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                content = f.read()

            if 'train:' in content and 'val:' in content and 'nc:' in content:
                self.info.append(f"✅ {yaml_path} - Valid")
            else:
                self.errors.append(f"❌ {yaml_path} - Configurare incompletă")
        else:
            self.warnings.append(f"⚠️  {yaml_path} lipsește")
            self._create_sample_yaml()

    def _create_sample_yaml(self):
        """Creează un fișier data_video.yaml de exemplu."""
        yaml_content = """# YOLO Dataset Configuration
# Proiect 9 - Control Adaptiv Faruri

train: datasets/train/images
val: datasets/valid/images

# Număr de clase (ajustează după nevoie)
nc: 1

# Numele claselor (ajustează după nevoie)
names: ['headlight']

# SAU pentru detecție vehicule generală:
# nc: 4
# names: ['car', 'truck', 'bus', 'motorcycle']
"""

        try:
            os.makedirs('datasets', exist_ok=True)
            with open('datasets/data.yaml', 'w') as f:
                f.write(yaml_content)
            self.info.append("✅ Creat datasets/data_video.yaml template")
        except Exception as e:
            self.errors.append(f"❌ Nu pot crea data_video.yaml: {e}")

    def create_missing_directories(self):
        """Creează directoarele lipsă."""
        print("\n🔧 Creare Directoare Lipsă...")

        dirs_to_create = [
            'datasets/train/images',
            'datasets/train/labels',
            'datasets/valid/images',
            'datasets/valid/labels',
            'video_test',
            'output_data',
            'analysis_output',
            'test_results'
        ]

        for dir_path in dirs_to_create:
            try:
                os.makedirs(dir_path, exist_ok=True)
                if not os.path.exists(dir_path):
                    self.info.append(f"✅ Creat: {dir_path}/")
            except Exception as e:
                self.errors.append(f"❌ Eroare creare {dir_path}/: {e}")

    def print_summary(self):
        """Afișează rezumatul verificărilor."""
        print("\n" + "=" * 70)
        print(" " * 20 + "📊 REZUMAT VERIFICARE")
        print("=" * 70)

        # Info
        if self.info:
            print("\n✅ INFORMAȚII:")
            for msg in self.info:
                print(f"  {msg}")

        # Warnings
        if self.warnings:
            print("\n⚠️  AVERTISMENTE:")
            for msg in self.warnings:
                print(f"  {msg}")

        # Errors
        if self.errors:
            print("\n❌ ERORI:")
            for msg in self.errors:
                print(f"  {msg}")

        # Status final
        print("\n" + "=" * 70)
        if not self.errors:
            if not self.warnings:
                print("🎉 PROIECT GATA DE RULARE!")
            else:
                print("✅ PROIECT FUNCȚIONAL (cu avertismente minore)")
        else:
            print("❌ NECESITĂ ATENȚIE - Rezolvă erorile de mai sus")
        print("=" * 70 + "\n")

    def run_full_check(self):
        """Rulează toate verificările."""
        print("=" * 70)
        print(" " * 15 + "🔍 VERIFICARE SETUP PROIECT 🔍")
        print("=" * 70)

        self.check_python_version()
        self.check_dependencies()
        self.create_missing_directories()
        self.check_directory_structure()
        self.check_data_yaml()
        self.check_video_files()
        self.check_trained_model()

        self.print_summary()

        # Sugestii
        if self.errors or self.warnings:
            print("\n💡 PAȘI URMĂTORI:\n")

            if any('pip install' in e for e in self.errors):
                print("1. Instalează dependințele:")
                print("   pip install -r requirements.txt\n")

            if any('imagini' in w for w in self.warnings):
                print("2. Adaugă imagini în datasets/train/images/")
                print("   - Descarcă un dataset de pe Roboflow")
                print("   - SAU folosește propriile imagini\n")

            if any('video' in str(self.warnings).lower() for w in self.warnings):
                print("3. Adaugă un fișier .mp4 în video_test/")
                print("   - Poate fi orice video cu mașini pe drum\n")

            if any('model' in w.lower() for w in self.warnings):
                print("4. Antrenează modelul:")
                print("   - Rulează celulele de antrenament din notebook")
                print("   - SAU folosește un script de antrenament\n")


def main():
    """Funcție principală."""
    checker = ProjectSetup()
    checker.run_full_check()

    # Întrebare finală
    print("\n" + "=" * 70)
    print("Vrei să rulezi sistemul acum? (necesită video și model)")
    print("=" * 70)
    print("\nComenzile disponibile:")
    print("  python main.py           - Rulează sistemul principal")
    print("  python data_analysis.py  - Analizează datasetul")
    print("  python model_testing.py  - Testează modelul")
    print()


if __name__ == "__main__":
    main()