# Classificateur de Mélanome

## Ce qui a été fait

### 1. Composants supprimés
- ✅ Temperature Scaling (classe TemperatureScaler)
- ✅ Fonctions de calibration (fit_temperature)
- ✅ ECE (Expected Calibration Error)
- ✅ Score de Brier
- ✅ Diagrammes de fiabilité
- ✅ Tout le code lié à la calibration

### 2. Composants ajoutés
- ✅ Support du modèle ConvNeXt (intégré dans base_model.py)
- ✅ Implémentation de GradCAM pour l'interprétabilité
- ✅ Script de visualisation GradCAM

### 3. Structure unifiée
- ✅ Séparation des responsabilités en modules : config, data, models, training, evaluation, interpretability
- ✅ Suppression de tout le code de notebooks
- ✅ Création de scripts Python propres
- ✅ Suppression de tous les commentaires du code

## Structure des fichiers

```
melanoma-classifier/
│
├── config/
│   └── config.py                 # Tous les paramètres de configuration
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                # Classe MelanomaDataset
│   └── data_loader.py            # Fonction load_ham10000_data
│
├── models/
│   ├── __init__.py
│   ├── base_model.py             # MelanomaClassifier (supporte tous les modèles)
│   ├── senet.py                  # Implémentation SENet
│   └── losses.py                 # FocalLoss
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                # Boucle d'entraînement, optimiseurs
│   └── augmentation.py           # Transformations Albumentations
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                # Métriques d'évaluation (sans calibration)
│   └── visualization.py          # Fonctions de visualisation
│
├── interpretability/
│   ├── __init__.py
│   └── gradcam.py                # Implémentation GradCAM
│
├── utils/
│   └── __init__.py
│
├── train.py                      # Script d'entraînement principal
├── evaluate.py                   # Script d'évaluation
├── visualize_gradcam.py          # Script de visualisation GradCAM
├── requirements.txt              # Dépendances
└── README.md                     # Documentation
```

## Modèles supportés

1. **ResNet50** - Définir `MODEL_NAME = 'resnet50'` dans la config
2. **SENet** - Définir `MODEL_NAME = 'senet'` dans la config
3. **EfficientNet** - Définir `MODEL_NAME = 'efficientnet'` dans la config
4. **ConvNeXt** - Définir `MODEL_NAME = 'convnext'` dans la config (NOUVEAU)
5. **VGG16** - Définir `MODEL_NAME = 'vgg16'` dans la config

## Fonctionnalités clés

### Fonctionnalités d'entraînement
- Focal Loss pour le déséquilibre de classes
- Taux d'apprentissage discriminatifs (différents LR selon les couches)
- Phase de préchauffage (entraînement de la tête uniquement en premier)
- Phase de fine-tuning (entraînement du modèle complet)
- Arrêt précoce (Early stopping)
- Planification du taux d'apprentissage

### Métriques d'évaluation (simplifiées)
- Exactitude (Accuracy)
- Précision
- Rappel (Recall)
- Score F1
- ROC-AUC
- Matrice de confusion
- Courbe Précision-Rappel

### Interprétabilité
- Cartes de chaleur GradCAM
- Explications visuelles des prédictions du modèle
- Compatible avec toutes les architectures

## Exemples d'utilisation

### 1. Entraîner un modèle

```bash
# Par défaut (ResNet50)
python train.py

# Pour d'autres modèles, modifier config/config.py :
# MODEL_NAME = 'senet'  # ou 'convnext', 'efficientnet', 'vgg16'
```

### 2. Évaluer un modèle

```bash
# Avec recherche automatique du seuil
python evaluate.py --checkpoint output/checkpoints/best_model.pth --model resnet50

# Avec un seuil personnalisé
python evaluate.py --checkpoint output/checkpoints/best_model.pth --model resnet50 --threshold 0.6
```

### 3. Générer des visualisations GradCAM

```bash
# Générer 20 visualisations depuis le jeu de test
python visualize_gradcam.py \
    --checkpoint output/checkpoints/best_model.pth \
    --model resnet50 \
    --num_samples 20 \
    --split test

# Visualiser depuis le jeu de validation
python visualize_gradcam.py \
    --checkpoint output/checkpoints/best_model.pth \
    --model convnext \
    --num_samples 10 \
    --split val \
    --output_dir ./output/gradcam_val
```

## Ce qui a changé par rapport au code original

### Depuis le notebook ResNet
- Suppression de tout le code spécifique aux notebooks
- Suppression de la calibration / temperature scaling
- Simplification des métriques
- Conservation de la logique d'entraînement principale
- Migration vers une structure modulaire

### Depuis le notebook SENet
- Extraction de l'architecture SENet vers models/senet.py
- Intégration dans le pipeline d'entraînement unifié
- Suppression du code de calibration
- Standardisation avec les autres modèles

### Depuis model.py
- Division en plusieurs modules
- Suppression du temperature scaling
- Suppression des métriques de calibration (ECE, Brier, fiabilité)
- Conservation de la Focal Loss
- Conservation des taux d'apprentissage discriminatifs
- Conservation du pipeline d'augmentation

## Nouveautés

### Support de ConvNeXt
- Ajouté dans models/base_model.py
- Utilise l'implémentation ConvNeXt de torchvision
- Supporte les poids pré-entraînés
- Compatible avec toutes les fonctionnalités d'entraînement

### GradCAM
- Nouveau module d'interprétabilité
- Génère des cartes d'attention
- Montre ce que le modèle "regarde"
- Aide au débogage et à la mise en confiance
- Compatible avec toutes les architectures

## Prochaines étapes

1. Placer les données HAM10000 dans le répertoire `./data/`
2. Modifier `config/config.py` pour choisir le modèle et les hyperparamètres
3. Exécuter `python train.py` pour entraîner
4. Exécuter `python evaluate.py` pour évaluer
5. Exécuter `python visualize_gradcam.py` pour interpréter

## Dépendances

Toutes les dépendances sont dans requirements.txt :
- numpy
- pandas
- torch
- torchvision
- Pillow
- opencv-python
- tqdm
- matplotlib
- seaborn
- scikit-learn
- albumentations
