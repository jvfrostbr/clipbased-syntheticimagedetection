import torch
import numpy as np
from PIL import Image
import cv2
import os
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

class CLIPSegModel:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print("🎨 Carregando CLIPSeg (Segmentação Visual)...")
        try:
            self.seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
            self.seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
            self.seg_model.eval()
        except Exception as e:
            print(f"❌ Erro ao baixar CLIPSeg: {e}")
            raise e

    def _generate_segmentation(self, image, prompts):
        """
        Gera as máscaras cruas usando o CLIPSeg.
        """
        inputs = self.seg_processor(
            text=prompts, 
            images=[image] * len(prompts), 
            padding=True, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.seg_model(**inputs)
        
        preds = outputs.logits
        
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(0)
            
        masks = torch.sigmoid(preds).cpu().numpy()
        
        w, h = image.size
        final_mask = np.zeros((h, w), dtype=np.float32)
        
        for mask in masks:
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            mask_resized = cv2.resize(mask, (w, h))
            final_mask = np.maximum(final_mask, mask_resized)
            
        return final_mask

    def generate_defect_overlay(self, image_path, prompts, overlay_color="auto", prompt_index=0):
        """
        Gera a máscara, aplica Filtro Bilateral perceptor de bordas,
        calcula a cor neon adaptativa mais distante e salva o overlay.
        """
        os.makedirs("outputs/defect_maps", exist_ok=True)
        image = Image.open(image_path).convert("RGB")
        img_np = np.array(image)
        
        print(f"   >>> Gerando Segmentação para: {prompts}")
        defect_map = self._generate_segmentation(image, prompts)

        # --- Normalização Min-Max ---
        defect_map_min = np.min(defect_map)
        defect_map_max = np.max(defect_map)
        if defect_map_max > defect_map_min:
            defect_map = (defect_map - defect_map_min) / (defect_map_max - defect_map_min)
        else:
            defect_map = np.zeros_like(defect_map)
        
        # binarização
        defect_map_bin = (defect_map > 0.35).astype(np.float32)
        
        # aplicando filtro Bilateral para suavizar a máscara mantendo as bordas nítidas
        mask_8u = (defect_map_bin * 255).astype(np.uint8)
        defect_map_smooth = cv2.bilateralFilter(mask_8u, d=9, sigmaColor=75, sigmaSpace=75)
        alpha_mask = defect_map_smooth.astype(np.float32) / 255.0

        # seleção de cor adaptativa baseada na média dos pixels afetados
        color_mask = np.zeros_like(img_np)
        
        if overlay_color == "auto":
            # Isola os pixels sob a ativação real da máscara
            pixels_afetados = img_np[defect_map_smooth > 100]
            cor_media = pixels_afetados.mean(axis=0) if len(pixels_afetados) > 0 else np.array([0, 0, 0])
            
            # Paleta de Alta Visibilidade Forense (Canais RGB puros)
            paleta_neon = [
                np.array([255, 0, 0]),    # Vermelho Neon
                np.array([0, 255, 0]),    # Verde Neon
                np.array([0, 0, 255]),    # Azul Neon
                np.array([255, 255, 0]),  # Amarelo Neon
                np.array([255, 0, 255]),  # Magenta Neon
                np.array([0, 255, 255])   # Ciano Neon
            ]

            # Seleção baseada na maior Distância Euclidiana contra o fundo médio afetado
            melhor_cor = max(paleta_neon, key=lambda cor: np.linalg.norm(cor_media - cor))
            color_mask[:, :] = melhor_cor
        else:
            # Fallbacks manuais passados por parâmetro string
            if overlay_color == "green":
                color_mask[:, :, 1] = 255
            elif overlay_color == "blue":
                color_mask[:, :, 2] = 255
            else: 
                color_mask[:, :, 0] = 255

        # Geração do overlay 
        img_float = img_np.astype(np.float32) / 255.0
        mask_float = color_mask.astype(np.float32) / 255.0
        alpha = alpha_mask[:, :, None]
        
        # Mistura alpha ponderada com proteção contra lavagem de contraste
        overlay = (mask_float * alpha * 0.5) + (img_float * (1.0 - (alpha * 0.2)))
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        
        # Converte para BGR e salva no Kaggle
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        base = os.path.basename(image_path)
        name_without_ext = os.path.splitext(base)[0]
        
        overlay_path = f"outputs/defect_maps/{name_without_ext}_defect_{prompt_index}.png"
        cv2.imwrite(overlay_path, overlay_bgr)
        
        return overlay_path