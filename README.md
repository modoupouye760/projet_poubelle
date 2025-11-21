# 🎯 Application Streamlit - Classification d'Images Multi-Modèles

Application web interactive pour comparer les performances de 3 modèles de classification d'images entraînés sur le dataset Caltech101.

## 🚀 Modèles Disponibles

1. **CNN Baseline** - Architecture personnalisée avec 3 blocs convolutionnels
2. **MobileNetV2 Transfer Learning** - Features pré-entraînées ImageNet gelées  
3. **EfficientNetB0 Transfer Learning** - Architecture avancée (ou fallback ResNet50)

## 📋 Fonctionnalités

### 🖼️ Upload d'Image Unique
- Upload direct d'images (JPG, PNG, WEBP)
- Prédictions simultanées des 3 modèles
- Comparaison visuelle des confidences
- Top 3 prédictions par modèle

### 🌐 Traitement par Lot via JSON
- Upload de fichier JSON avec URLs d'images
- Traitement automatique de multiples images
- Export des résultats en CSV
- Visualisation des distributions de prédictions

### 📈 Analyse Comparative
- Métriques de performance des modèles
- Graphiques interactifs
- Tableau récapitulatif des performances

## 🛠️ Installation

### 1. Prérequis
Assurez-vous d'avoir Python 3.8+ installé.

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```

### 3. Préparation des modèles
Avant de lancer l'application, vous devez :

1. **Exécuter le notebook** `Classe-Exercice_1_CNN_vs_TransferLearning_Caltech101.ipynb` 
2. **Exécuter la cellule de sauvegarde** (section 8) qui créera le dossier `saved_models/` avec :
   - `cnn_baseline.h5`
   - `mobilenetv2_transfer.h5` 
   - `efficientnet_transfer.h5`
   - `metadata.json`

### 4. Lancement de l'application
```bash
streamlit run streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

## 📁 Structure des Fichiers

```
CNN_projet_poubelle/
├── streamlit_app.py              # Application Streamlit principale
├── requirements.txt              # Dépendances Python
├── example_images.json          # Exemple de fichier JSON pour tests
├── saved_models/                # Modèles sauvegardés (créé après notebook)
│   ├── cnn_baseline.h5
│   ├── mobilenetv2_transfer.h5
│   ├── efficientnet_transfer.h5
│   └── metadata.json
└── Classe-Exercice_1_CNN_vs_TransferLearning_Caltech101.ipynb
```

## 🔧 Format JSON pour Upload par Lot

Le fichier JSON doit suivre cette structure :

```json
{
  "images": [
    {
      "name": "Description de l'image",
      "url": "https://example.com/image1.jpg"
    },
    {
      "name": "Autre image",
      "url": "https://example.com/image2.jpg"
    }
  ]
}
```

Un fichier d'exemple `example_images.json` est fourni pour tester.

## 📊 Classes Supportées (Caltech101 - Sous-ensemble)

L'application supporte 20 classes du dataset Caltech101 :
- accordion, airplanes, anchor, ant, barrel, bass, beaver, binocular, bonsai, brain, 
- brontosaurus, buddha, butterfly, camera, cannon, car_side, ceiling_fan, cellphone, chair, chandelier

## ⚡ Fonctionnalités Avancées

### Cache Intelligent
- Les modèles sont chargés une seule fois grâce au cache Streamlit
- Optimisation des performances pour les prédictions répétées

### Interface Responsive
- Design adaptatif pour desktop et mobile
- Graphiques interactifs avec Plotly
- Interface intuitive en onglets

### Gestion d'Erreurs
- Validation des uploads
- Messages d'erreur informatifs  
- Fallback en cas d'échec de chargement

## 🐛 Résolution de Problèmes

### Erreur "metadata.json non trouvé"
➡️ Exécutez d'abord la cellule de sauvegarde dans le notebook

### Erreur de chargement des modèles
➡️ Vérifiez que tous les fichiers .h5 sont présents dans `saved_models/`

### Images ne se chargent pas depuis URLs
➡️ Vérifiez que les URLs sont accessibles et pointent vers des images valides

### Performances lentes
➡️ Les modèles Transfer Learning peuvent être lents au premier chargement

## 📈 Utilisation Recommandée

1. **Test Rapide** : Utilisez l'onglet "Upload Image" avec une image locale
2. **Évaluation Poussée** : Utilisez l'onglet "URLs JSON" pour tester sur plusieurs images
3. **Analyse** : Consultez l'onglet "Analyse Comparative" pour comprendre les performances

## 🎓 Contexte Pédagogique

Cette application illustre :
- La comparaison CNN custom vs Transfer Learning
- L'impact des architectures pré-entraînées
- Les métriques de performance en classification d'images
- Le déploiement de modèles ML avec Streamlit

---
🎯 **Projet réalisé dans le cadre du cours de Deep Learning - Comparaison CNN vs Transfer Learning**