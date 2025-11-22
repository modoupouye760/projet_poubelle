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
    page_title="Vision Artificielle - Détection de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 CSS custom - Design moderne premium
custom_css = """
<style>
    /* Reset et fond principal */
    .main {
        background: linear-gradient(135deg, #0f1f0f 0%, #1a331a 100%);
        background-attachment: fixed;
    }
    
    /* Container principal élargi */
    .main .block-container {
        background: rgba(15, 31, 15, 0.95);
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.4);
        max-width: 95%;
        backdrop-filter: blur(10px);
        border: 1px solid #2d5a2d;
    }
    
    /* Header principal premium */
    .main-header {
        background: linear-gradient(135deg, #1e3d1e 0%, #2d5a2d 100%);
        color: white;
        padding: 5rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 3rem;
        position: relative;
        overflow: hidden;
        border: 1px solid #3d6b3d;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 30px 30px;
        animation: float 25s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-30px, -30px) rotate(360deg); }
    }
    
    .main-title {
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-shadow: 4px 4px 8px rgba(0,0,0,0.4);
        position: relative;
        color: #f0fff0;
        background: linear-gradient(135deg, #e8f5e8 0%, #a8d8a8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 1.8rem;
        opacity: 0.95;
        font-weight: 300;
        position: relative;
        color: #d0e8d0;
        letter-spacing: 1px;
    }
    
    /* Barre d'outils supérieure */
    .toolbar {
        background: rgba(30, 61, 30, 0.9);
        backdrop-filter: blur(15px);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid #3d6b3d;
        color: white;
    }
    
    /* Boutons premium */
    .stButton>button {
        background: linear-gradient(135deg, #2d5a2d 0%, #3d7a3d 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 14px 28px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(45, 90, 45, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton>button:hover::before {
        left: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(45, 90, 45, 0.7);
        background: linear-gradient(135deg, #3d7a3d 0%, #4d8a4d 100%);
    }
    
    /* Bouton de téléchargement */
    .download-btn {
        background: linear-gradient(135deg, #5a2d8c 0%, #7a3dad 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(90, 45, 140, 0.5) !important;
    }
    
    .download-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 30px rgba(90, 45, 140, 0.7) !important;
        background: linear-gradient(135deg, #7a3dad 0%, #8a4dbd 100%) !important;
    }
    
    /* Cartes de contenu premium */
    .content-card {
        background: linear-gradient(135deg, #1e3d1e 0%, #2d5a2d 100%);
        color: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border: 1px solid #3d6b3d;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .content-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4d8a4d, #3d7a3d, #2d5a2d);
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }
    
    /* Zone d'upload premium */
    .upload-section {
        background: linear-gradient(135deg, #1e3d1e 0%, #2d5a2d 100%);
        color: white;
        border: 2px dashed #4d8a4d;
        border-radius: 20px;
        padding: 5rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 30% 70%, rgba(77, 138, 77, 0.1) 0%, transparent 50%);
    }
    
    .upload-section:hover {
        background: linear-gradient(135deg, #2d5a2d 0%, #3d7a3d 100%);
        border-color: #5d9a5d;
        transform: scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.35);
    }
    
    /* Badges de résultats premium */
    .detection-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3d7a3d 0%, #5d9a5d 100%);
        color: white;
        padding: 12px 28px;
        border-radius: 25px;
        margin: 10px;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(61, 122, 61, 0.5);
        font-size: 1.1rem;
        border: 1px solid #4d8a4d;
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar-container {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #3d6b3d;
        backdrop-filter: blur(10px);
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #ff4757 0%, #ffa502 50%, #2ed573 100%);
        height: 14px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Statistiques premium */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 3rem 0;
        text-align: center;
        gap: 2rem;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #2d5a2d 0%, #3d7a3d 100%);
        color: white;
        padding: 2rem 1.5rem;
        border-radius: 20px;
        flex: 1;
        box-shadow: 0 12px 30px rgba(45, 90, 45, 0.5);
        border: 1px solid #3d6b3d;
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stat-item:hover {
        transform: translateY(-5px);
    }
    
    .stat-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4d8a4d, #5d9a5d);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        display: block;
        color: #f0fff0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        color: #d0e8d0;
        letter-spacing: 0.5px;
    }
    
    /* Textes dans les cartes */
    .content-card h1, .content-card h2, .content-card h3, 
    .content-card h4, .content-card h5, .content-card h6 {
        color: #f0fff0 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .content-card p, .content-card div {
        color: #d0e8d0 !important;
    }
    
    /* Section des fonctionnalités */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #1e3d1e 0%, #2d5a2d 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #3d6b3d;
        transition: all 0.3s ease;
        box-shadow: 0 15px 35px rgba(0,0,0,0.25);
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.35);
    }
    
    .feature-icon {
        font-size: 3.5rem;
        margin-bottom: 1.5rem;
        display: block;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ---------------------------------------
# 🧠 CHARGEMENT DU MODÈLE YOLO
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
        st.success("✅ Modèle YOLO initialisé avec succès!")
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
    <div class="main-title">🤖 Vision Artificielle</div>
    <div class="main-subtitle">Détection Intelligente • Analyse en Temps Réel • Technologie Avancée</div>
</div>
""", unsafe_allow_html=True)

# Avertissements de dépendances
if not CV2_AVAILABLE:
    st.warning("""
    ⚠️ **Module OpenCV manquant** 
    - Fonctionnalités d'affichage limitées
    - L'analyse principale reste opérationnelle
    """)

if not ULTRALYTICS_AVAILABLE:
    st.error("""
    ❌ **Module Ultralytics requis**
    - Impossible d'initialiser les modèles de vision
    - Vérifiez l'installation des composants
    """)

# ---------------------------------------
# 📥 SECTION CONFIGURATION DU MODÈLE
# ---------------------------------------
st.markdown("<div class='content-card'>", unsafe_allow_html=True)
st.markdown("### 🧠 Configuration du Système de Vision")

if model is None:
    st.error("""
    🔧 **Configuration requise**
    
    Pour activer le système :
    1. Téléchargez le modèle de détection
    2. Placez le fichier dans le dossier `models/`
    3. Assurez-vous qu'il se nomme `best.pt`
    """)
else:
    st.success("✅ **Système de vision opérationnel**")
    
    # Informations sur le modèle
    col_info, col_download = st.columns([2, 1])
    
    with col_info:
        st.markdown("""
        ### 📊 Spécifications Techniques
        - **Architecture**: YOLOv8 Optimisé
        - **Domaine**: Vision par ordinateur
        - **Statut**: 🟢 Système actif
        - **Performance**: Détection en temps réel
        """)
        
        # Affichage des classes détectables
        if hasattr(model, 'names'):
            st.markdown("### 🎯 Objets Reconnaissables")
            classes = list(model.names.values())
            classes_text = " • ".join(classes)
            st.markdown(f"**Domaines de détection:** {classes_text}")
    
    with col_download:
        st.markdown("### 📦 Gestion des Modèles")
        
        # Bouton de téléchargement du modèle actuel
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                model_data = f.read()
            
            st.download_button(
                label="💾 Exporter le Modèle",
                data=model_data,
                file_name="model_vision.pt",
                mime="application/octet-stream",
                help="Téléchargez le modèle de vision actuel",
                use_container_width=True,
                key="download_model"
            )
            
            # Informations sur le modèle
            file_size = len(model_data) / (1024 * 1024)  # Taille en MB
            st.info(f"**Poids du modèle:** {file_size:.1f} MB")
        
        st.markdown("---")
        st.markdown("### 🔗 Ressources")
        st.markdown("""
        **Modèles de référence:**
        - [YOLOv8 Nano](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
        - [YOLOv8 Small](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
        - [YOLOv8 Medium](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)
        """)
    
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 📸 SECTION ACQUISITION D'IMAGE
# ---------------------------------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center;'>
    <h3 style='color: #f0fff0; margin-bottom: 1.5rem; font-size: 2.2rem;'>📷 Zone d'Acquisition</h3>
    <p style='color: #d0e8d0; font-size: 1.2rem; line-height: 1.6;'>
        Importez une image pour analyse automatique<br>
        <span style='font-size: 1rem; opacity: 0.9;'>Formats supportés: JPG, JPEG, PNG, BMP</span>
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_img = st.file_uploader(
    " ",
    type=["jpg", "jpeg", "png", "bmp"],
    key="main_uploader",
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🖼️ PROCESSUS D'ANALYSE
# ---------------------------------------
if uploaded_img and ULTRALYTICS_AVAILABLE and model is not None:
    # Layout principal pour visualisation
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("### 🖼️ Image Source")
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Image importée pour analyse", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur de traitement: {e}")
            uploaded_img = None
        st.markdown("</div>", unsafe_allow_html=True)

    # Bouton d'analyse central
    st.markdown("<div style='text-align: center; margin: 3rem 0;'>", unsafe_allow_html=True)
    analyze = st.button(
        "🚀 Démarrer l'Analyse IA", 
        type="primary", 
        use_container_width=True,
        help="Lance le processus de détection automatique"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze:
        with st.spinner("🔍 **Scan en cours...** Le système analyse l'image"):
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
                    st.markdown("### 📈 Résultats de l'Analyse")
                    
                    if CV2_AVAILABLE:
                        try:
                            # Annotation avec visualisation
                            annotated = r.plot()
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption="🟢 Détections par vision artificielle", use_container_width=True)
                        except Exception as e:
                            st.warning("⚠️ Visualisation avancée non disponible")
                            st.image(image, caption="Image source (mode basique)", use_container_width=True)
                    else:
                        st.image(image, caption="Image source (module vision non disponible)", use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

                # Métriques de performance
                dets = getattr(r, "boxes", None)
                if dets and len(dets) > 0:
                    st.markdown("<div class='stats-container'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="stat-item">
                        <span class="stat-number">{len(dets)}</span>
                        <span class="stat-label">Objets Identifiés</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">{max(len(dets), 1)}</span>
                        <span class="stat-label">Analyses Réussies</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-number">YOLOv8</span>
                        <span class="stat-label">Moteur IA</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Détails analytiques
                    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown("### 🔬 Analyse Détaillée")
                    
                    for i, box in enumerate(dets, start=1):
                        cls_idx = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls_idx] if hasattr(model, "names") else str(cls_idx)
                        
                        # Affichage avec métriques de confiance
                        conf_percent = int(conf * 100)
                        st.markdown(f"""
                        <div class="confidence-bar-container">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                <span class="detection-badge">🎯 Analyse #{i} • {cls_name.upper()}</span>
                                <strong style="font-size: 1.4rem; color: #f0fff0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{conf_percent}%</strong>
                            </div>
                            <div class="confidence-bar" style="width: {conf_percent}%;"></div>
                            <div style="text-align: center; color: #d0e8d0; font-size: 1rem; margin-top: 10px; font-weight: 500;">
                Niveau de certitude de l'intelligence artificielle
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("🎯 Aucun objet détecté dans le cadre d'analyse")
            else:
                st.error("❌ Aucune donnée d'analyse générée")

elif uploaded_img and (not ULTRALYTICS_AVAILABLE or model is None):
    st.error("❌ Système de vision non opérationnel - Analyse impossible")

else:
    # Section présentation des capacités
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("### 💫 Fonctionnalités du Système")
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <h4>Intelligence Artificielle</h4>
            <p>Technologie de vision par ordinateur avancée pour une détection précise</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <h4>Temps Réel</h4>
            <p>Analyse et traitement instantanés des images importées</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <h4>Précision</h4>
            <p>Détection avec métriques de confiance détaillées</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Guide d'utilisation
    col_guide1, col_guide2, col_guide3 = st.columns(3)
    
    with col_guide1:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <div style='font-size: 3.5rem; margin-bottom: 1.5rem;'>1️⃣</div>
            <h4 style='color: #f0fff0; margin-bottom: 1rem;'>Configuration</h4>
            <p style='color: #d0e8d0; line-height: 1.6;'>Vérifiez la disponibilité du modèle de vision artificielle</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <div style='font-size: 3.5rem; margin-bottom: 1.5rem;'>2️⃣</div>
            <h4 style='color: #f0fff0; margin-bottom: 1rem;'>Acquisition</h4>
            <p style='color: #d0e8d0; line-height: 1.6;'>Importez l'image à analyser via la zone dédiée</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_guide3:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem;'>
            <div style='font-size: 3.5rem; margin-bottom: 1.5rem;'>3️⃣</div>
            <h4 style='color: #f0fff0; margin-bottom: 1rem;'>Analyse</h4>
            <p style='color: #d0e8d0; line-height: 1.6;'>Lancez le processus de détection automatique</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------
# 🏁 PIED DE PAGE
# ---------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #d0e8d0; padding: 4rem 1rem; background: linear-gradient(135deg, rgba(30,61,30,0.8) 0%, rgba(45,90,45,0.6) 100%); border-radius: 20px; margin-top: 3rem;'>
    <h3 style='color: #f0fff0; margin-bottom: 1.5rem; font-size: 2rem;'>Système de Vision Artificielle</h3>
    <p style='font-size: 1.2rem; margin-bottom: 1rem; line-height: 1.6;'>🚀 Propulsé par YOLOv8 & Streamlit</p>
    <p style='font-size: 1rem; opacity: 0.8; line-height: 1.5;'>Technologie de détection avancée • Intelligence artificielle embarquée • Performance optimisée</p>
</div>
""", unsafe_allow_html=True)
