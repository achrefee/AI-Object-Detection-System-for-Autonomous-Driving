# 📘 Guide Complet — Système de Détection d'Objets pour la Conduite Autonome

> **Projet PFE** | Février 2026  
> Ce guide vous accompagne de la préparation des données BDD100K jusqu'à l'entraînement YOLO.

---

## 📅 Calendrier Résumé

| Jour | Étape | Objectif |
|------|-------|----------|
| 1 | Structure du projet | ✅ **Fait** — Répertoires et scripts créés |
| 1–2 | Télécharger BDD100K | Obtenir images + labels depuis Berkeley |
| 2 | Convertir en YOLO | Transformer les labels JSON → format YOLO |
| 3 | Filtrer les classes | Valider les 11 classes cibles |
| 4 | Nettoyer le dataset | Supprimer les fichiers invalides |
| 5 | Équilibrer le dataset | Corriger les classes sous-représentées |
| 6 | Diviser le dataset | Train 70% / Val 20% / Test 10% |
| 7+ | Entraîner YOLO | Lancer l'entraînement sur Kaggle |

---

## 🧩 ÉTAPE 1 — Structure du Projet ✅ FAIT

```
autonomous_vision/
├── data/
│   ├── bdd100k/          ← BDD100K extrait (images + labels JSON)
│   ├── raw/              ← Converti en format YOLO
│   ├── processed/        ← Après filtrage et nettoyage
│   ├── train/            ← 70% pour l'entraînement
│   ├── val/              ← 20% pour la validation
│   └── test/             ← 10% pour les tests
├── scripts/              ← Scripts de traitement
├── dataset.yaml          ← Config YOLO (11 classes)
└── README.md
```

---

## 🧩 ÉTAPE 2 — Télécharger BDD100K (Jour 1–2)

### Pourquoi BDD100K ?
- **100 000 images dashcam** de vraies scènes de conduite
- Couvre **11 classes** pertinentes dont les feux tricolores par couleur
- Labels de haute qualité avec bounding boxes
- C'est **le** dataset de référence pour la conduite autonome

### Téléchargement

1. Créer un compte sur [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/)
2. Télécharger :
   - **Images** : `bdd100k_images_100k.zip` (~6.8 Go)
   - **Labels** : Labels de détection (format JSON)

### Extraction

Extraire les ZIPs dans `data/bdd100k/` :

```bash
cd autonomous_vision

# Créer le répertoire
mkdir -p data/bdd100k

# Extraire les images et labels dans data/bdd100k/
# Structure attendue après extraction :
#   data/bdd100k/images/100k/train/*.jpg
#   data/bdd100k/images/100k/val/*.jpg
#   data/bdd100k/labels/det_20/det_train.json
#   data/bdd100k/labels/det_20/det_val.json
```

### Les 11 classes du projet

| ID | Classe | Catégorie | Source BDD100K |
|----|--------|-----------|----------------|
| 0 | `car` | Véhicule | car ✅ |
| 1 | `truck` | Véhicule | truck ✅ |
| 2 | `bus` | Véhicule | bus ✅ |
| 3 | `motorcycle` | Véhicule | motorcycle ✅ |
| 4 | `bicycle` | Véhicule | bicycle ✅ |
| 5 | `pedestrian` | Usager Vulnérable | pedestrian ✅ |
| 6 | `cyclist` | Usager Vulnérable | rider ✅ |
| 7 | `traffic_light_red` | Signalisation | traffic light (red) ✅ |
| 8 | `traffic_light_green` | Signalisation | traffic light (green) ✅ |
| 9 | `traffic_light_yellow` | Signalisation | traffic light (yellow) ✅ |
| 10 | `traffic_sign` | Signalisation | traffic sign ✅ |

---

## 🧩 ÉTAPE 3 — Convertir en Format YOLO (Jour 2)

### Prérequis

```bash
pip install Pillow
```

### Commande à exécuter

```bash
cd autonomous_vision

# 🧪 Test rapide (500 images — vérifier que tout fonctionne)
python scripts/convert_bdd100k.py --max-images 500

# 🚀 Conversion complète (toutes les images)
python scripts/convert_bdd100k.py
```

### Options disponibles

| Option | Description |
|--------|-------------|
| `--max-images 500` | Limiter le nombre d'images (pour tester) |
| `--split train` | Convertir uniquement le split train |
| `--split val` | Convertir uniquement le split validation |
| `--bdd-dir data/bdd100k` | Chemin vers BDD100K (défaut) |

### Ce que fait le script

1. 🔍 Détecte automatiquement la structure des fichiers BDD100K
2. 📖 Lit les labels JSON (supporte le format det_20 et l'ancien format)
3. 🔄 Convertit les coordonnées `(x1, y1, x2, y2)` → YOLO `(cx, cy, w, h)`
4. 🎨 Classe les feux tricolores par couleur (rouge/vert/jaune)
5. 📁 Sauvegarde dans `data/raw/images/` et `data/raw/labels/`

### Format YOLO (généré automatiquement)

```
# <class_id> <x_center> <y_center> <width> <height>
# Toutes les valeurs sont normalisées entre 0 et 1
0 0.4532 0.6210 0.1200 0.2500
5 0.7800 0.5500 0.0400 0.1800
7 0.1200 0.1500 0.0300 0.0600
```

---

## 🧩 ÉTAPE 4 — Filtrer les Classes (Jour 3)

```bash
python scripts/filter_classes.py --raw-dir data/raw --out-dir data/processed
```

Valide les class IDs et copie uniquement les données avec des annotations valides.

---

## 🧩 ÉTAPE 5 — Nettoyer le Dataset (Jour 4)

```bash
# Prévisualiser (sans rien supprimer)
python scripts/clean_dataset.py --data-dir data/processed --dry-run

# Nettoyer
python scripts/clean_dataset.py --data-dir data/processed
```

Supprime : labels vides, images sans label, labels sans image, images corrompues.

---

## 🧩 ÉTAPE 6 — Équilibrer le Dataset (Jour 5)

```bash
# Analyser (ne modifie rien)
python scripts/balance_dataset.py --data-dir data/processed --analyze-only

# Équilibrer (augmenter les classes rares)
python scripts/balance_dataset.py --data-dir data/processed --min-objects 1000
```

Duplique et augmente les images des classes sous-représentées (luminosité, contraste, flou, flip).

---

## 🧩 ÉTAPE 7 — Diviser le Dataset (Jour 6)

```bash
python scripts/split_dataset.py --src-dir data/processed --out-dir data --copy
```

| Split | Pourcentage | Répertoire |
|-------|-------------|------------|
| Train | 70% | `data/train/` |
| Val | 20% | `data/val/` |
| Test | 10% | `data/test/` |

> [!IMPORTANT]
> Utilisez `--copy` pour garder les données originales dans `data/processed/`.

---

## 🧩 ÉTAPE 8 — Entraîner avec YOLO (Phase 3)

### Sur Kaggle Notebook (GPU gratuit)

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
```

---

## Phase 3 : Entraînement du Modèle (Kaggle)

### Étape 1 : Préparer le Dataset pour Kaggle

Après avoir exécuté tout le pipeline (conversion → filtrage → nettoyage → équilibrage → split), vous aurez :

```
data/
├── train/
│   ├── images/    (70% des données)
│   └── labels/
├── val/
│   ├── images/    (20% des données)
│   └── labels/
└── test/
    ├── images/    (10% des données)
    └── labels/
```

**Créer un ZIP pour Kaggle :**

```bash
# Compresser le dataset final + dataset.yaml
cd autonomous_vision
zip -r driving-dataset.zip data/train data/val data/test dataset.yaml
```

Ensuite, uploadez `driving-dataset.zip` comme **Kaggle Dataset** sur [kaggle.com/datasets](https://www.kaggle.com/datasets).

### Étape 2 : Entraîner sur Kaggle

1. Créer un **nouveau Notebook** sur Kaggle
2. **Settings** → Accelerator → **GPU T4 x2** (ou P100)
3. **Settings** → Internet → **ON**
4. Ajouter votre dataset au notebook
5. Copier-coller le contenu de `notebooks/kaggle_training.py` dans une cellule

Le script effectue automatiquement :

| Phase | Détail | Époques |
|-------|--------|---------|
| **Phase 1** | Transfer Learning (backbone gelé) | 10 |
| **Phase 2** | Fine-Tuning (tous les layers) | 100 |
| **Évaluation** | Métriques sur le test set | — |
| **Export** | Conversion en ONNX | — |

> ⚠️ **Sessions Kaggle** : Max **12 heures** par session. Le script sauvegarde des checkpoints toutes les 10 époques. Pour reprendre :
> ```python
> model = YOLO("/kaggle/input/previous-run/last.pt")
> results = model.train(resume=True)
> ```

### Étape 3 : Récupérer le Modèle

Après l'entraînement, cliquez **"Save Version"** → **"Save & Run All"** sur Kaggle pour persister les sorties.

Téléchargez `best.pt` depuis l'onglet **Output** du notebook et placez-le dans :

```
autonomous_vision/
└── weights/
    └── best.pt
```

---

## Phase 3 : Inférence en Temps Réel

### Lancer le Pipeline

```bash
cd autonomous_vision

# Webcam (caméra par défaut)
python -m src.pipeline.realtime_pipeline --model weights/best.pt

# Fichier vidéo
python -m src.pipeline.realtime_pipeline --source driving_video.mp4 --model weights/best.pt

# Avec sauvegarde vidéo
python -m src.pipeline.realtime_pipeline --source video.mp4 --model weights/best.pt --output result.mp4

# Avec estimation de profondeur MiDaS (plus précis, plus lent)
python -m src.pipeline.realtime_pipeline --source video.mp4 --model weights/best.pt --midas

# CPU uniquement
python -m src.pipeline.realtime_pipeline --source video.mp4 --model weights/best.pt --device cpu
```

**Contrôles :**
- `Q` ou `ESC` : Quitter
- Le HUD affiche : détections, distances, zones de risque, FPS, action en cours

### Options CLI

| Argument | Défaut | Description |
|----------|--------|-------------|
| `--source` | `0` | Fichier vidéo ou index caméra |
| `--model` | `yolov8s.pt` | Chemin vers les poids YOLO |
| `--config` | `configs` | Dossier de configuration |
| `--device` | `cuda` | `cuda` ou `cpu` |
| `--confidence` | `0.35` | Seuil de confiance |
| `--output` | — | Chemin vidéo de sortie |
| `--midas` | off | Activer MiDaS depth |
| `--no-display` | off | Désactiver l'affichage |

---

## Calibration Caméra (Optionnel)

Pour améliorer la précision de l'estimation de distance, calibrez votre caméra :

```bash
# Depuis des images de damier
python scripts/camera_calibration.py --images calibration_images/ --board 9x6

# Depuis la caméra en direct (appuyer ESPACE pour capturer)
python scripts/camera_calibration.py --camera 0 --board 9x6
```

Les paramètres calibrés sont sauvegardés dans `configs/camera_params.yaml`.

---

## 📋 Checklist Résumée

### Phase 2 : Dataset
- [x] Créer la structure du projet
- [x] Créer les scripts de traitement
- [x] Configurer `dataset.yaml` (11 classes BDD100K)
- [ ] Télécharger BDD100K depuis bdd-data.berkeley.edu
- [ ] Extraire les ZIPs dans `data/bdd100k/`
- [ ] Tester la conversion (`convert_bdd100k.py --max-images 500`)
- [ ] Lancer la conversion complète (`convert_bdd100k.py`)
- [ ] Exécuter `filter_classes.py`
- [ ] Exécuter `clean_dataset.py`
- [ ] Exécuter `balance_dataset.py`
- [ ] Exécuter `split_dataset.py`

### Phase 3 : Entraînement & Inférence
- [ ] Compresser et uploader le dataset sur Kaggle
- [ ] Lancer `notebooks/kaggle_training.py` sur Kaggle (GPU T4 x2)
- [ ] Télécharger `best.pt` → `weights/best.pt`
- [ ] Tester : `python -m src.pipeline.realtime_pipeline --model weights/best.pt`
- [ ] (Optionnel) Calibrer la caméra : `python scripts/camera_calibration.py`

---

*Guide créé pour le Projet PFE — Détection d'Objets par IA pour la Conduite Autonome*
