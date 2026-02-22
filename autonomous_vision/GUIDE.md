# 📘 Guide Complet — Système de Détection d'Objets pour la Conduite Autonome

> **Projet PFE** | Février 2026  
> Ce guide vous accompagne jour par jour, de la préparation des données jusqu'à l'entraînement YOLO.

---

## 📅 Calendrier Résumé

| Jour | Étape | Objectif |
|------|-------|----------|
| 1 | Structure du projet | ✅ **Fait** — Répertoires et scripts créés |
| 1–2 | Télécharger COCO 2017 | Obtenir images + labels via script automatique |
| 3 | Filtrer les classes | Garder uniquement les 18 classes cibles |
| 4 | Nettoyer le dataset | Supprimer les fichiers invalides |
| 5 | Équilibrer le dataset | Corriger les classes sous-représentées |
| 6 | Diviser le dataset | Train 70% / Val 20% / Test 10% |
| 6 | Configurer dataset.yaml | ✅ **Fait** — Prêt pour l'entraînement YOLO |
| 7+ | Compléter les classes manquantes | Ajouter GTSRB, données custom, etc. |

---

## 🧩 ÉTAPE 1 — Structure du Projet ✅ FAIT

```
autonomous_vision/
├── data/
│   ├── raw/              ← Données téléchargées brutes
│   ├── processed/        ← Après filtrage et nettoyage
│   ├── train/            ← 70% pour l'entraînement
│   ├── val/              ← 20% pour la validation
│   └── test/             ← 10% pour les tests
├── scripts/              ← Scripts de traitement
├── dataset.yaml          ← Config YOLO
└── README.md
```

---

## 🧩 ÉTAPE 2 — Télécharger le Dataset COCO 2017 (Jour 1–2)

### Pourquoi COCO ?
- **330 000 images** dont beaucoup de scènes de conduite
- 8 des 18 classes cibles disponibles directement
- **Téléchargement automatique** via script (pas d'inscription manuelle)
- Labels de détection de haute qualité

### Prérequis — Installer les dépendances

```bash
pip install fiftyone ultralytics Pillow
```

> [!NOTE]
> FiftyOne télécharge COCO depuis les serveurs officiels. La première exécution peut prendre du temps (~20 Go).

### Commande à exécuter

```bash
cd autonomous_vision

# 🧪 Test rapide d'abord (500 images — vérifier que tout fonctionne)
python scripts/download_coco.py --max-samples 500

# 🚀 Téléchargement complet (toutes les images pertinentes)
python scripts/download_coco.py
```

### Options disponibles

| Option | Description |
|--------|-------------|
| `--max-samples 500` | Limiter le nombre d'images (pour tester) |
| `--split train` | Télécharger uniquement le split train |
| `--split validation` | Télécharger uniquement le split validation |
| `--split both` | Télécharger les deux (par défaut) |

### Ce que fait le script

1. 📥 Télécharge COCO 2017 via FiftyOne
2. 🔍 Filtre uniquement les 8 classes pertinentes pour la conduite
3. 📦 Exporte en format YOLO (`.txt`)
4. 🔄 Remappe les class IDs vers notre numérotation (0–17)
5. 📁 Organise tout dans `data/raw/images/` et `data/raw/labels/`

### Résultat attendu
```
📥 Downloading COCO 2017 — train split
✅ Downloaded 82783 samples
   After filtering: ~45000 samples with target classes
📦 Exporting to YOLO format → data/coco_export
🔄 Remapped 45000 label files to target class IDs
📁 Organized into data/raw:
   Images: 45000
   Labels: 45000
🎉 COCO Download Complete!
```

### Classes couvertes par COCO

| ✅ Couvertes (8/18) | ❌ À compléter plus tard (10/18) |
|--------------------|----------------------------------|
| car, truck, bus | cyclist |
| motorcycle, bicycle | traffic_light_green, traffic_light_yellow |
| pedestrian | speed_limit_sign, yield_sign, no_entry_sign |
| traffic_light (→ red), stop_sign | road_barrier, cone, pothole, crosswalk |

> [!TIP]
> **Alternative : BDD100K** — Si vous préférez des images spécifiquement de conduite,
> vous pouvez télécharger BDD100K manuellement depuis [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/).
> Dans ce cas, changez `CLASS_MAPPING = BDD100K_MAPPING` dans `filter_classes.py`.

### Format YOLO (généré automatiquement)

Chaque fichier `.txt` dans `data/raw/labels/` contient :
```
# <class_id> <x_center> <y_center> <width> <height>
# Toutes les valeurs sont normalisées entre 0 et 1
0 0.4532 0.6210 0.1200 0.2500
5 0.7800 0.5500 0.0400 0.1800
```

---

## 🧩 ÉTAPE 3 — Filtrer les Classes (Jour 3)

### Objectif
Garder uniquement les **18 classes cibles** et supprimer tout le reste.

### Les 18 classes du projet

| ID | Classe | Catégorie | Source |
|----|--------|-----------|--------|
| 0 | `car` | Véhicule | COCO ✅ |
| 1 | `truck` | Véhicule | COCO ✅ |
| 2 | `bus` | Véhicule | COCO ✅ |
| 3 | `motorcycle` | Véhicule | COCO ✅ |
| 4 | `bicycle` | Véhicule | COCO ✅ |
| 5 | `pedestrian` | Usager Vulnérable | COCO ✅ |
| 6 | `cyclist` | Usager Vulnérable | ⏳ BDD100K / Custom |
| 7 | `traffic_light_red` | Signalisation | COCO ⚠️ (à raffiner par couleur) |
| 8 | `traffic_light_green` | Signalisation | ⏳ À raffiner depuis COCO |
| 9 | `traffic_light_yellow` | Signalisation | ⏳ À raffiner depuis COCO |
| 10 | `stop_sign` | Signalisation | COCO ✅ |
| 11 | `speed_limit_sign` | Signalisation | ⏳ GTSRB / Mapillary |
| 12 | `yield_sign` | Signalisation | ⏳ GTSRB / Mapillary |
| 13 | `no_entry_sign` | Signalisation | ⏳ GTSRB / Mapillary |
| 14 | `road_barrier` | Obstacle | ⏳ Custom / CARLA |
| 15 | `cone` | Obstacle | ⏳ Custom / CARLA |
| 16 | `pothole` | Obstacle | ⏳ Custom / Kaggle |
| 17 | `crosswalk` | Route | ⏳ Custom |

### Commande à exécuter

```bash
cd autonomous_vision
python scripts/filter_classes.py --raw-dir data/raw --out-dir data/processed
```

### Résultat attendu
```
📂 Found 70000 label files in data/raw/labels
✅ Filtering complete!
   Kept:    58000 images
   Dropped: 12000 images (no valid objects)
📊 Class distribution:
   car                 : 120000
   pedestrian          :  45000
   truck               :  12000
   ...
```

---

## 🧩 ÉTAPE 4 — Nettoyer le Dataset (Jour 4)

### Objectif
Supprimer les fichiers problématiques :
- ❌ Labels vides (0 octets)
- ❌ Labels sans image correspondante
- ❌ Images sans label correspondant
- ❌ Images corrompues / illisibles
- ❌ Labels avec format invalide

### Commande à exécuter

```bash
# D'abord, prévisualiser ce qui sera supprimé (sans rien supprimer)
python scripts/clean_dataset.py --data-dir data/processed --dry-run

# Si tout semble correct, nettoyer pour de vrai
python scripts/clean_dataset.py --data-dir data/processed
```

### Prérequis
```bash
pip install Pillow    # Pour vérifier les images corrompues
```

---

## 🧩 ÉTAPE 5 — Équilibrer le Dataset (Jour 5)

### Le problème
```
car          : 120000  ← Beaucoup trop
stop_sign    :    500  ← Pas assez !
cone         :     50  ← Le modèle va ignorer cette classe
```

Si le dataset est déséquilibré, **le modèle ignore les classes rares**.

### Commande à exécuter

```bash
# Étape 1 : Analyser la distribution (ne modifie rien)
python scripts/balance_dataset.py --data-dir data/processed --analyze-only

# Étape 2 : Équilibrer (augmenter les classes rares à minimum 1000 objets)
python scripts/balance_dataset.py --data-dir data/processed --min-objects 1000
```

### Ce que fait le script
1. Compte les objets par classe
2. Pour chaque classe sous le seuil :
   - Duplique des images contenant cette classe
   - Applique des augmentations simples (luminosité, contraste, flou, flip)
   - Ajuste les labels en conséquence

---

## 🧩 ÉTAPE 6 — Diviser le Dataset (Jour 6)

### Commande à exécuter

```bash
python scripts/split_dataset.py --src-dir data/processed --out-dir data --copy
```

### Résultat

| Split | Pourcentage | Répertoire |
|-------|-------------|------------|
| Train | 70% | `data/train/images/` + `data/train/labels/` |
| Val | 20% | `data/val/images/` + `data/val/labels/` |
| Test | 10% | `data/test/images/` + `data/test/labels/` |

> [!IMPORTANT]
> Utilisez `--copy` pour garder les données originales dans `data/processed/` en backup.
> Sans `--copy`, les fichiers sont **déplacés** (pas de backup).

---

## 🧩 ÉTAPE 7 — Vérifier dataset.yaml (Jour 6)

Le fichier `dataset.yaml` est déjà configuré avec les 18 classes :

```yaml
path: data
train: train/images
val: val/images
test: test/images

nc: 18
names:
  0: car
  1: truck
  2: bus
  # ... (18 classes au total)
  17: crosswalk
```

> Ce fichier sera utilisé directement par Ultralytics YOLO pour l'entraînement.

---

## 🧩 ÉTAPE 8 — Compléter les Classes Manquantes (Jour 7+)

COCO couvre 8 des 18 classes. Voici comment compléter les 10 restantes :

### Sources recommandées

| Classes manquantes | Dataset | Lien |
|-------------------|---------|------|
| `cyclist` | **BDD100K** (rider class) | [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/) |
| `traffic_light_green/yellow` | Raffiner depuis les détections COCO | Script custom à créer |
| `speed_limit_sign`, `yield_sign`, `no_entry_sign` | **GTSRB** | [benchmark.ini.rub.de](https://benchmark.ini.rub.de/) |
| `road_barrier`, `cone` | **CARLA Simulator** ou collection personnelle | [carla.org](https://carla.org/) |
| `pothole` | **Kaggle Pothole Dataset** | Chercher "pothole detection" sur Kaggle |
| `crosswalk` | **Collection personnelle** | Dashcam footage |

### Processus pour chaque source supplémentaire

1. Télécharger le dataset
2. Convertir les labels en format YOLO
3. Remapper les class IDs vers notre numérotation (0–17)
4. Copier dans `data/processed/images/` et `data/processed/labels/`
5. Re-exécuter `clean_dataset.py` et `balance_dataset.py`
6. Re-exécuter `split_dataset.py`

---

## 🧩 ÉTAPE 9 — Entraîner avec YOLO (Phase 3 du projet)

### Sur Kaggle Notebook (GPU gratuit)

```python
from ultralytics import YOLO

# Charger le modèle pré-entraîné
model = YOLO("yolov8s.pt")

# Lancer l'entraînement
results = model.train(
    data="dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
)
```

> Voir le rapport complet (`AI_Object_Detection_System_Report.md`) pour les paramètres détaillés et la configuration Kaggle.

---

## 📋 Checklist Résumée

- [x] Créer la structure du projet
- [x] Créer les scripts de traitement
- [x] Créer le script de téléchargement COCO
- [x] Configurer `dataset.yaml` (18 classes)
- [ ] Installer les dépendances (`pip install fiftyone ultralytics Pillow`)
- [ ] Tester le téléchargement (`download_coco.py --max-samples 500`)
- [ ] Lancer le téléchargement complet (`download_coco.py`)
- [ ] Exécuter `filter_classes.py`
- [ ] Exécuter `clean_dataset.py`
- [ ] Exécuter `balance_dataset.py`
- [ ] Exécuter `split_dataset.py`
- [ ] Télécharger les datasets complémentaires (GTSRB, etc.)
- [ ] Lancer l'entraînement YOLO sur Kaggle

---

## ❓ Besoin d'Aide ?

| Si vous êtes bloqué sur... | Demandez-moi... |
|---------------------------|-----------------|
| Erreur avec FiftyOne | Copiez-collez l'erreur |
| Ajouter les données BDD100K | "Change le mapping pour BDD100K" |
| Convertir GTSRB | "Crée un script pour convertir GTSRB en YOLO" |
| Entraînement Kaggle | "Crée le notebook Kaggle d'entraînement" |
| Erreurs dans les scripts | Copiez-collez l'erreur |
| Module de distance | "Crée le module d'estimation de distance" |

---

*Guide créé pour le Projet PFE — Détection d'Objets par IA pour la Conduite Autonome*
