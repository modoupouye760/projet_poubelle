# app.py
import streamlit as st
import os
import numpy as np
from PIL import Image
import io

# Configuration pour éviter les problèmes OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Import sécurisé d'OpenCV
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ OpenCV non disponible: {e}")
    CV2_AVAILABLE = False

# Import sécurisé d'Ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Ultralytics non disponible: {e}")
    ULTRALYTICS_AVAILABLE = False

# ---------------------------------------
# 🎨 CONFIG INTERFACE MODERNE
# ---------------------------------------
st.set_page_config(
    page_title="Détection Intelligente de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 CSS custom - Design moderne avec cartes vert foncé
custom_css = """
<style>
    /* Reset et fond principal */
    .main {
        background: linear-gradient(135deg, #1a2f1a 0%, #2d4a2d 100%);
        background-attachment: fixed;
    }
    
    /* Container principal élargi */
    .main .block-container {
        background: #1a2f1a;
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
        max-width: 95%;
    }
    
    /* Header principal centré */
    .main-header {
        background: linear-gradient(135deg, #2d4a2d 0%, #3d6b3d 100%);
        color: white;
        padding: 4rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 2px solid #4a7c4a;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        position: relative;
        color: #e8f5e8;
    }
    
    .main-subtitle {
        font-size: 1.6rem;
        opacity: 0.95;
        font-weight: 300;
        position: relative;
        color: #c8e6c8;
    }
    
    /* Barre d'outils supérieure */
    .toolbar {
        background: rgba(45, 74, 45, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border: 1px solid #4a7c4a;
        color: white;
    }
    
    /* Boutons modernes */
    .stButton>button {
        background: linear-gradient(135deg, #4a7c4a 0%, #5d995d 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 12px 25px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 124, 74, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(74, 124, 74, 0.6);
        background: linear-gradient(135deg, #5d995d 0%, #6bb06b 100%);
    }
    
    /* Bouton de téléchargement */
    .download-btn {
        background: linear-gradient(135deg, #6b46c1 0%, #805ad5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 12px 25px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(107, 70, 193, 0.4) !important;
    }
    
    .download-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(107, 70, 193, 0.6) !important;
        background: linear-gradient(135deg, #805ad5 0%, #9f7aea 100%) !important;
    }
    
    /* Cartes de contenu en VERT FONCÉ */
    .content-card {
        background: linear-gradient(135deg, #2d4a2d 0%, #3d6b3d 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 2px solid #4a7c4a;
        margin-bottom: 2rem;
        transition: transform 0.3s ease;
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    /* Zone d'upload stylisée */
    .upload-section {
        background: linear-gradient(135deg, #2d4a2d 0%, #3d6b3d 100%);
        color: white;
        border: 3px dashed #5d995d;
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .upload-section:hover {
        background: linear-gradient(135deg, #3d6b3d 0%, #4a7c4a 100%);
        border-color: #6bb06b;
        transform: scale(1.02);
    }
    
    /* Badges de résultats */
    .detection-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4a7c4a 0%, #6bb06b 100%);
        color: white;
        padding: 10px 25px;
        border-radius: 25px;
        margin: 8px;
        font-weight: 600;
        box-shadow: 0 6px 20px rgba(74, 124, 74, 0.4);
        font-size: 1.1rem;
        border: 1px solid #5d995d;
    }
    
    .confidence-bar-container {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #4a7c4a;
        backdrop-filter: blur(10px);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff6b6b 0%, #ffd93d 50%, #6bcf7f 100%);
        height: 12px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Statistiques */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        text-align: center;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #4a7c4a 0%, #5d995d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        min-width: 150px;
        box-shadow: 0 8px 25px rgba(74, 124, 74, 0.4);
        border: 1px solid #5d995d;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        display: block;
        color: #e8f5e8;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        color: #c8e6c8;
    }
    
    /* Textes dans les cartes */
    .content-card h1, .content-card h2, .content-card h3, 
    .content-card h4, .content-card h5, .content-card h6 {
        color: #e8f5e8 !important;
    }
    
    .content-card p, .content-card div {
        color: #c8e6c8 !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------
# 🧠 CHARGEMENT DU MODEL YOLO
# ---------------------------------------
MODEL_PATH = "models/best.pt"

def ensure_models_directory():
    """Crée le dossier models s'il n'existe pas"""
    os.makedirs("models", exist_ok=True)
    return os.path.exists("models")

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        model = YOLO(path)
        st.success("✅ Modèle YOLO chargé avec succès!")
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

# Initialisation
ensure_models_directory()
model = load_model() if ULTRALYTICS_AVAILABLE else None

# ---------------------------------------
# 🖥️ HEADER PRINCIPAL
# ---------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🗑️ Détection Intelligente</div>
    <div class="main-subtitle">IA Avancée · Détection en Temps Réel · Classification Automatique</div>
</div>
""", unsafe_allow_html=True)

# Avertissements de dépendances
if not CV2_AVAILABLE:
    st.warning("""
    ⚠️ **OpenCV non disponible** 
    - L'affichage des images annotées sera limité
    - La détection fonctionne normalement
    """)

if not ULTRALYTICS_AVAILABLE:
    st.error("""
    ❌ **Ultralytics non disponible**
    - Impossible de charger les modèles YOLO
    - Vérifiez l'installation des dépendances
    """)

# ---------------------------------------
# 📥 SECTION TÉLÉCHARGEMENT DU MODÈLE
# ---------------------------------------
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("### 🚀 Configuration du Modèle IA")

if model is None:
    st.error("""
    ❌ **Modèle introuvable**
    
    Pour utiliser l'application :
    1. Placez votre fichier `best.pt` dans le dossier `models/`
    2. Le modèle doit s'appeler `best.pt` et être placé dans le dossier `models/`
    """)
else:
    st.success("✅ **Modèle chargé avec succès!**")
    
    # Informations sur le modèle
    col_info, col_download = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        ### 📋 Informations du Modèle
        - **Type**: YOLOv8
        - **Fonction**: Détection de poubelles
        - **Statut**: ✅ Opérationnel
        """)
        
        # Affichage des classes détectables
        if hasattr(model, 'names'):
            st.markdown("### 🏷️ Classes Détectables")
            classes = list(model.names.values())
            classes_text = ", ".join(classes)
            st.markdown(f"**Objets reconnus:** {classes_text}")
    
    with col_download:
        st.markdown("### 📥 Téléchargement")
        
        # Bouton de téléchargement du modèle actuel
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model_data = f.read()
            
            st.download_button(
                label="💾 Télécharger le Modèle",
                data=model_data,
                file_name="best.pt",
                mime="application/octet-stream",
                help="Téléchargez le modèle YOLO de détection de poubelles",
                use_container_width=True,
                key="download_model"
            )
            
            # Informations sur le modèle
            file_size = len(model_data) / (1024 * 1024)  # Taille en MB
            st.info(f"**Taille du modèle:** {file_size:.1f} MB")
        
        st.markdown("---")
        st.markdown("### 🔗 Modèles Pré-entraînés")
        st.markdown("""
        **Modèles YOLOv8 officiels:**
        - [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
        - [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
        - [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)
        """)
    
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 📸 SECTION UPLOAD D'IMAGE
# ---------------------------------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### 📸 Analyse d'Image")
st.markdown("""
<div style='text-align: center;'>
    <h3 style='color: #e8f5e8; margin-bottom: 1rem;'>⬆️ Glissez-déposez votre image ici</h3>
    <p style='color: #c8e6c8; font-size: 1.1rem;'>Formats supportés: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

uploaded_img = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png"],
    key="main_uploader",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🖼️ AFFICHAGE DES RÉSULTATS
# ---------------------------------------
if uploaded_img and ULTRALYTICS_AVAILABLE and model is not None:
    # Layout principal pour images
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Originale")
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Image source uploadée", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur de chargement: {e}")
            uploaded_img = None
        st.markdown("</div>", unsafe_allow_html=True)

    # Bouton d'analyse centré
    st.markdown("<div style='text-align: center; margin: 2rem 0;'>", unsafe_allow_html=True)
    analyze = st.button(
        "🚀 Lancer l'Analyse IA", 
        type="primary", 
        use_container_width=True
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze:
        with st.spinner("🔍 **Analyse en cours...** L'IA scanne l'image"):
            # Conversion et prédiction
            img_array = np.array(image)
            
            try:
                results = model.predict(img_array, conf=0.25, imgsz=640)
            except Exception as e:
                st.error(f"❌ Erreur d'analyse: {e}")
                results = None

            if results and len(results) > 0:
                r = results[0]

                # Affichage résultats dans colonne 2
                with col2:
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Résultats de Détection")
                    
                    if CV2_AVAILABLE:
                        try:
                            # Tentative d'annotation avec OpenCV
                            annotated = r.plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption="🟢 Détections YOLOv8", use_container_width=True)
                        except Exception as e:
                            st.warning("⚠️ Annotation OpenCV non disponible")
                            st.image(image, caption="Image originale (annotation non disponible)", use_container_width=True)
                    else:
                        st.image(image, caption="Image originale (OpenCV non disponible)", use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                # Statistiques de détection
                dets = getattr(r, "boxes", None)
                if dets and len(dets) > 0:
                    st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="stat-item">
                        <span class="stat-number">{len(dets)}</span>
                        <span class="stat-label">Poubelles Détectées</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">{max(len(dets), 1)}</span>
                        <span class="stat-label">Analyses Effectuées</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">YOLOv8</span>
                        <span class="stat-label">Modèle IA</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Détails des détections
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Détails des Analyses")
                    
                    for i, box in enumerate(dets, start=1):
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_idx] if hasattr(model, "names") else str(cls_idx)
                        
                        # Affichage avec barre de confiance
                        conf_percent = int(conf * 100)
                        st.markdown(f"""
                        <div class="confidence-bar-container">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span class="detection-badge">🔍 Détection #{i} • {cls_name.upper()}</span>
                                <strong style="font-size: 1.3rem; color: #e8f5e8;">{conf_percent}%</strong>
                            </div>
                            <div class="confidence-bar" style="width: {conf_percent}%;"></div>
                            <div style="text-align: center; color: #c8e6c8; font-size: 0.9rem; margin-top: 5px;">
                                Niveau de confiance de l'IA
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("❌ Aucune poubelle détectée dans l'image")
            else:
                st.error("❌ Aucun résultat d'analyse obtenu")

elif uploaded_img and (not ULTRALYTICS_AVAILABLE or model is None):
    st.error("❌ Modèle non disponible - Impossible d'analyser l'image")

else:
    # Section d'instructions quand aucune image n'est uploadée
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("### 💡 Guide d'Utilisation")
    
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    
    with col_guide1:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>1️⃣</div>
            <h4 style='color: #e8f5e8;'>Modèle Pré-configuré</h4>
            <p style='color: #c8e6c8;'>Utilisez le modèle YOLO pré-configuré pour la détection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide2:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>2️⃣</div>
            <h4 style='color: #e8f5e8;'>Import d'Image</h4>
            <p style='color: #c8e6c8;'>Sélectionnez une image contenant une ou plusieurs poubelles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide3:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>3️⃣</div>
            <h4 style='color: #e8f5e8;'>Analyse IA</h4>
            <p style='color: #c8e6c8;'>Lancez la détection et visualisez les résultats en temps réel</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🏁 FOOTER
# ---------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #c8e6c8; padding: 3rem 1rem;'>
    <h3 style='color: #e8f5e8; margin-bottom: 1rem;'>Détection Intelligente de Poubelles</h3>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>🚀 Propulsé par YOLOv8 & Streamlit</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Système de détection et classification automatique • IA de pointe</p>
</div>
""", unsafe_allow_html=True)
