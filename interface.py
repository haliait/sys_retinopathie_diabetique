# ============================================================
# INTERFACE CLINIQUE DE DÉTECTION DE LA RÉTINOPATHIE DIABÉTIQUE
# Architecture : ResNet-18 fine-tuné (Phase 4)
# Framework    : Gradio
# Résolution   : 224x224 pixels
# ============================================================

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import io
import warnings
warnings.filterwarnings('ignore')

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ── Configuration ────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE    = 224       # résolution d'entraînement ResNet-18
NUM_CLASSES = 5

print(f"Device utilisé : {DEVICE}")

# ── Nommage clinique des classes ─────────────────────────────
CLASS_NAMES = {
    0: 'No DR (Pas de rétinopathie)',
    1: 'Mild DR (Rétinopathie légère)',
    2: 'Moderate DR (Rétinopathie modérée)',
    3: 'Severe DR (Rétinopathie sévère)',
    4: 'Proliferative DR (Rétinopathie proliférante)'
}

CLASS_COLORS = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#8e44ad']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

RECOMMENDATIONS = {
    0: "✅ Aucune rétinopathie détectée. Contrôle annuel recommandé.",
    1: "🟡 Rétinopathie légère. Contrôle ophtalmologique dans les 12 mois.",
    2: "🟠 Rétinopathie modérée. Consultation ophtalmologique recommandée sous 6 mois.",
    3: "🔴 Rétinopathie sévère. Référence URGENTE en ophtalmologie (< 1 mois).",
    4: "🚨 Rétinopathie proliférante. Référence IMMÉDIATE. Risque de cécité."
}

# ── Préprocessing médical (identique Phase 2) ────────────────
def crop_image_from_gray(img, tol=7):
    """
    Supprime le fond noir périphérique des rétinographies.
    Le fond noir ne contient aucune information clinique et
    biaiserait la prédiction si conservé.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        check = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        if check.shape[0] == 0 or check.shape[1] == 0:
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]

def ben_graham_preprocessing(img, sigmaX=30):
    """
    Rehaussement du contraste local (Ben Graham, Kaggle 2015).
    Formule : 4*img - 4*GaussianBlur(img, sigma=30) + 128
    Amplifie les hautes fréquences spatiales (lésions fines)
    et supprime les variations d'illumination globale.
    """
    return cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE sur le canal L de l'espace LAB.
    Égalisation adaptative du contraste par tuiles 8x8,
    appliquée uniquement à la luminance pour préserver
    l'information colorimétrique clinique (couleur des lésions).
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe   = cv2.createCLAHE(clipLimit=clip_limit,
                                tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_clahe, a, b]),
                         cv2.COLOR_LAB2RGB)

def preprocess_image_for_model(pil_image):
    """
    Pipeline complet de préprocessing médical :
    1. Conversion RGB
    2. Circle Crop (suppression fond noir)
    3. Resize 224x224
    4. Ben Graham (rehaussement contraste local)
    5. CLAHE (égalisation adaptative)
    6. Normalisation ImageNet + conversion tensor

    Retourne :
        tensor         : [1, 3, 224, 224] normalisé ImageNet
        img_preprocessed : np.array uint8 [224, 224, 3] pour affichage
        img_original    : np.array uint8 original pour référence
    """
    img          = np.array(pil_image.convert('RGB'))
    img_original = img.copy()

    # Étape 1 : Circle Crop
    img = crop_image_from_gray(img)

    # Étape 2 : Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Étape 3 : Ben Graham
    img = ben_graham_preprocessing(img)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Étape 4 : CLAHE
    img = apply_clahe(img)

    img_preprocessed = img.copy()  # sauvegarde pour affichage

    # Étape 5 : Normalisation ImageNet + tensor
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    tensor = transform(image=img)['image'].unsqueeze(0)

    return tensor, img_preprocessed, img_original

# ── Chargement du modèle ResNet-18 ───────────────────────────
def load_model(model_path):
    """
    Charge le modèle ResNet-18 fine-tuné depuis le checkpoint Phase 4.

    Architecture :
    - Backbone : ResNet-18 pré-entraîné ImageNet (sans tête originale)
    - in_features : 512 (sortie du layer4[-1] de ResNet-18)
    - Tête personnalisée : Dropout(0.3) -> Linear(512,512)
                           -> SiLU -> Dropout(0.2) -> Linear(512,5)

    La tête doit être IDENTIQUE à celle utilisée en Phase 4,
    sinon le chargement des poids échouera.
    """
    # Backbone sans tête de classification
    model = timm.create_model(
        'resnet18',
        pretrained=False,
        num_classes=0,        # supprime la tête originale
        global_pool='avg'
    )
    in_features = model.num_features  # 512 pour ResNet-18

    # Tête personnalisée — identique à Phase 4
    head = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES)
    )
    model.fc=head

    # Chargement des poids sauvegardés
    checkpoint = torch.load(model_path,
                             map_location=DEVICE,
                             weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ model_state_dict chargé")
        if 'kappa_par_phase' in checkpoint:
            kappa = checkpoint['kappa_par_phase']
            print(f"  κ Phase 1 : {kappa.get('Phase 1', 'N/A')}")
            print(f"  κ Phase 2 : {kappa.get('Phase 2', 'N/A')}")
            print(f"  κ Phase 3 : {kappa.get('Phase 3', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ State dict direct chargé")

    model = model.to(DEVICE)
    model.eval()
    print(f"✓ Modèle ResNet-18 prêt sur {DEVICE}")
    return model

# ── Génération de la heatmap Grad-CAM ────────────────────────
def generate_gradcam(model, input_tensor, predicted_class,
                      img_preprocessed):
    """
    Génère la heatmap Grad-CAM pour la classe prédite.

    Couche cible : model.layer4[-1]
    → Dernier bloc résiduel de ResNet-18.
    → Contient les features les plus sémantiques et discriminantes,
      idéales pour localiser les lésions rétiniennes.
    → Convention standard pour Grad-CAM sur architectures ResNet.

    Retourne :
        cam_image     : np.array RGB avec heatmap superposée
        grayscale_cam : np.array [H,W] valeurs [0,1]
    """
    # Couche cible = dernier bloc résiduel de ResNet-18
    target_layers = [model.layer4[-1]]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        targets       = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # Normalisation de l'image préprocessée pour superposition
    img_float = img_preprocessed.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_float, grayscale_cam,
                                   use_rgb=True)

    return cam_image, grayscale_cam

# ── Graphique des probabilités par classe ────────────────────
def create_probability_chart(probabilities):
    """
    Crée un barplot horizontal des probabilités softmax pour les 5 classes.
    Code couleur clinique : vert (sain) → violet (prolifératif).
    La classe prédite est encadrée en noir épais.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    classes = [
        f"C{i}: {list(CLASS_NAMES.values())[i].split('(')[0].strip()}"
        for i in range(NUM_CLASSES)
    ]

    bars = ax.barh(classes, probabilities,
                   color=CLASS_COLORS,
                   edgecolor='black', linewidth=0.5)

    for bar, prob in zip(bars, probabilities):
        ax.text(
            min(prob + 0.01, 0.92),
            bar.get_y() + bar.get_height() / 2,
            f'{prob * 100:.1f}%',
            va='center', ha='left',
            fontsize=10, fontweight='bold'
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Probabilité', fontsize=11)
    ax.set_title('Distribution des probabilités par stade DR',
                  fontweight='bold')
    ax.axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Encadrer la classe prédite
    pred_class = int(np.argmax(probabilities))
    bars[pred_class].set_edgecolor('black')
    bars[pred_class].set_linewidth(3)

    plt.tight_layout()

    # Conversion figure → image numpy pour Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    chart_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    chart_img = cv2.imdecode(chart_arr, cv2.IMREAD_COLOR)
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_BGR2RGB)
    plt.close()

    return chart_img

# ── Fonction principale de prédiction ────────────────────────
inference_model = None   # variable globale pour éviter de recharger

def predict_dr(input_image):
    """
    Fonction principale appelée par Gradio à chaque upload.

    Pipeline :
    1. Préprocessing médical (crop + Ben Graham + CLAHE + normalisation)
    2. Inférence ResNet-18 → probabilités softmax [5]
    3. Grad-CAM sur layer4[-1]
    4. Génération du graphique de probabilités
    5. Formatage du résultat clinique

    Retourne 5 outputs dans l'ordre des composants Gradio :
        cam_image, result_text, preprocessed_img,
        prob_chart, status_bar
    """
    global inference_model

    # Chargement paresseux du modèle (une seule fois)
    if inference_model is None:
        if MODEL_PATH and os.path.exists(MODEL_PATH):
            inference_model = load_model(MODEL_PATH)
        else:
            return (
                None,
                "⚠️ Modèle non chargé. Vérifier MODEL_PATH dans la cellule précédente.",
                None, None,
                "❌ Erreur : checkpoint introuvable."
            )

    if input_image is None:
        return (
            None,
            "*Veuillez uploader une image de fond d'œil.*",
            None, None,
            "En attente d'une image..."
        )

    try:
        # ── 1. Préprocessing ──────────────────────────────────
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)

        tensor, img_preprocessed, img_original = \
            preprocess_image_for_model(input_image)
        tensor = tensor.to(DEVICE)

        # ── 2. Inférence ──────────────────────────────────────
        with torch.no_grad():
            logits = inference_model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        predicted_class = int(np.argmax(probs))
        confidence      = float(probs[predicted_class]) * 100

        # ── 3. Grad-CAM ───────────────────────────────────────
        tensor_cam = tensor.clone().requires_grad_(True)
        cam_image, _ = generate_gradcam(
            inference_model, tensor_cam,
            predicted_class, img_preprocessed
        )

        # ── 4. Résultat clinique ──────────────────────────────
        stade_name     = CLASS_NAMES[predicted_class]
        recommendation = RECOMMENDATIONS[predicted_class]

        result_text = f"""
## 🔬 Résultat de l'Analyse

| Paramètre | Valeur |
|-----------|--------|
| **Stade détecté** | {stade_name} |
| **Confiance** | {confidence:.1f}% |
| **Modèle** | ResNet-18 (fine-tuné, κ=0.5263) |

**Recommandation clinique :**
{recommendation}

---
> ⚠️ **AVERTISSEMENT RÉGLEMENTAIRE**
> Ce système est un outil d'aide à la décision médicale (prototype académique).
> Il ne remplace pas le diagnostic d'un ophtalmologue certifié.
> Toute décision thérapeutique doit être validée par un professionnel de santé.
        """

        # ── 5. Graphique des probabilités ─────────────────────
        prob_chart = create_probability_chart(probs)

        status = (
            f"✓ Analyse terminée — "
            f"Stade {predicted_class} ({stade_name.split('(')[0].strip()}) "
            f"— Confiance : {confidence:.1f}%"
        )

        return cam_image, result_text, img_preprocessed, prob_chart, status

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        return (
            None,
            f"❌ Erreur lors de l'analyse :\n```\n{str(e)}\n```\n"
            f"Vérifier que l'image est une rétinographie valide.",
            None, None,
            f"❌ Erreur : {str(e)}"
        )

# ── Construction de l'interface Gradio ───────────────────────
def create_interface():
    """
    Construit l'interface Gradio pour les professionnels de santé.

    Composants :
    - Zone d'upload d'image (PNG, JPEG, TIFF)
    - Bouton d'analyse
    - Résultat clinique en Markdown
    - Image préprocessée (pipeline Phase 2 visible)
    - Carte Grad-CAM (explicabilité)
    - Graphique de probabilités par stade
    - Barre de statut
    - Section d'information sur le modèle (accordéon)
    """
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Détection de la Rétinopathie Diabétique — Génie Digital en Santé"
    ) as demo:

        # ── En-tête ──────────────────────────────────────────
        gr.Markdown("""
        # 👁️ Système de Détection Automatique de la Rétinopathie Diabétique
        ### Projet de Fin d'Année — 4ème Année Génie Digital en Santé

        **Instructions :**
        1. Télécharger une photographie du fond d'œil (rétinographie)
        2. Cliquer sur **Analyser l'image**
        3. Consulter le stade détecté, la carte d'activation et la recommandation

        > ⚠️ Prototype académique — résultats à valider par un ophtalmologue.
        """)

        # ── Ligne principale ─────────────────────────────────
        with gr.Row():

            # Colonne gauche : input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="📤 Image du fond d'œil (rétinographie)",
                    type="pil",
                    height=350
                )
                analyze_btn = gr.Button(
                    "🔍 Analyser l'image",
                    variant="primary",
                    size="lg"
                )
                gr.Markdown("""
                **Formats acceptés :** JPEG, PNG, TIFF

                **Conseils :**
                - Image nette et bien exposée
                - Rétine centrée dans l'image
                - Résolution minimale recommandée : 224×224 px
                """)

            # Colonne droite : résultat clinique
            with gr.Column(scale=2):
                result_text = gr.Markdown(
                    value="*En attente d'une image...*"
                )

        # ── Images préprocessée et Grad-CAM ──────────────────
        with gr.Row():
            with gr.Column():
                preprocessed_img = gr.Image(
                    label="🔧 Image préprocessée (Circle Crop + Ben Graham + CLAHE)",
                    height=280
                )
            with gr.Column():
                gradcam_img = gr.Image(
                    label="🔥 Carte Grad-CAM — Zones d'attention du modèle (layer4[-1])",
                    height=280
                )

        # ── Graphique des probabilités ────────────────────────
        with gr.Row():
            prob_chart = gr.Image(
                label="📊 Distribution des probabilités par stade DR",
                height=280
            )

        # ── Barre de statut ───────────────────────────────────
        status_bar = gr.Textbox(
            label="Statut",
            interactive=False,
            value="Prêt — En attente d'une image..."
        )

        # ── Accordéon d'information ───────────────────────────
        with gr.Accordion("ℹ️ Informations sur le modèle et le système",
                           open=False):
            gr.Markdown("""
            ### Spécifications du modèle

            | Paramètre | Valeur |
            |-----------|--------|
            | **Architecture** | ResNet-18 (He et al., CVPR 2016) |
            | **Sélection** | Benchmark Phase 3 (meilleur κ sur GPU T4 Colab) |
            | **Résolution d'entrée** | 224 × 224 pixels |
            | **Epochs entraînés** | 4 (contraintes GPU Google Colab T4) |
            | **Technique d'optimisation** | Progressive Unfreezing + Focal Loss (γ=2) + OneCycleLR |
            | **Inférence** | TTA 6 augmentations |
            | **Meilleur κ (val)** | 0.5263 (Phase 3 — fine-tuning complet) |
            | **Dataset** | APTOS 2019 — 3 662 images de fond d'œil |
            | **Explicabilité** | Grad-CAM sur layer4[-1] (Selvaraju et al., ICCV 2017) |

            ### Classes détectées

            | Code | Stade clinique | Couleur |
            |------|---------------|---------|
            | 0 | No DR — Pas de rétinopathie | 🟢 |
            | 1 | Mild DR — Rétinopathie légère | 🟡 |
            | 2 | Moderate DR — Rétinopathie modérée | 🟠 |
            | 3 | Severe DR — Rétinopathie sévère | 🔴 |
            | 4 | Proliferative DR — Rétinopathie proliférante | 🟣 |

            ### Équipe de développement
            Projet de Fin d'Année — 4ème Année Génie Digital en Santé
            """)

        # ── Connexion des événements ──────────────────────────
        analyze_btn.click(
            fn=predict_dr,
            inputs=[input_image],
            outputs=[
                gradcam_img,
                result_text,
                preprocessed_img,
                prob_chart,
                status_bar
            ]
        )

    return demo

print("✓ Interface définie — exécuter la cellule suivante pour lancer")
