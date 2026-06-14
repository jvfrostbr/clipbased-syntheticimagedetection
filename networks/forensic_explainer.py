import os
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image

from  segmentation_clipseg import CLIPSegModel 

class ForensicExplainer:
    def __init__(self, clip_base_model, preprocess_fn, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inicializando Módulo de Explicabilidade Forense no {self.device}...")

        # Reaproveita o seu CLIP Vit-L/14 para a extração de conceitos (CBM)
        self.clip_model = clip_base_model
        self.preprocess = preprocess_fn

        # Carrega os conceitos e as âncoras dos arquivos .txt
        self._load_configurations()

        # Instanciando a classe CLIPSegModel
        print("Acoplando o seu CLIPSegModel com Filtro Bilateral")
        self.segmentador = CLIPSegModel(device=self.device)

    def _load_configurations(self):
        """Lê os arquivos txt de conceitos e âncoras para memória."""
        self.concepts_eng = []      
        self.concepts_map = {}      
        self.visual_anchors = {}    

        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config") 

        try:
            with open(os.path.join(config_dir, "concepts.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if ";" in line:
                        eng, pt = line.strip().split(";")
                        self.concepts_eng.append(eng.strip())
                        self.concepts_map[eng.strip()] = pt.strip()
            print(f"✅ Carregados {len(self.concepts_eng)} conceitos.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar concepts.txt: {e}")

        try:
            with open(os.path.join(config_dir, "anchors.txt"), "r", encoding="utf-8") as f:
                for line in f:
                    if ";" in line:
                        key, target = line.strip().split(";")
                        self.visual_anchors[key.strip()] = target.strip()
            print(f"✅ Carregadas {len(self.visual_anchors)} âncoras visuais.")
        except Exception as e:
            print(f"⚠️ Erro ao carregar anchors.txt: {e}")

    def analisar_conceitos(self, image_path, classificacao_preliminar=None):
        """Extrai conceitos forenses da imagem usando CLIP, com limiar dinâmico."""
        conceitos_completos = self.concepts_eng + ["a high quality natural photograph"]

        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_tokens = open_clip.tokenize(conceitos_completos).to(self.device)

            threshold = 0.25 if classificacao_preliminar in ["a real photograph", 0] else 0.10

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = text_probs.cpu().numpy()[0]

            resultado = {}
            for i in range(len(self.concepts_eng)):
                if probs[i] > threshold: 
                    resultado[self.concepts_eng[i]] = float(probs[i])
            
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            print(f"⚠️ Erro na análise de conceitos: {e}")
            return {}

    def explain_decision(self, image_path, label_eng, overlay_color="auto"):
        """
        Intercepta as ativações do CBM e delega a geração do mapa para a sua classe.
        """
        # Executa a sua etapa padrão de análise via CLIP
        conceitos_eng = self.analisar_conceitos(image_path, classificacao_preliminar=label_eng)
        
        defect_maps_output = []

        if not conceitos_eng:
            print("⚠️ Nenhum conceito superou o threshold. Sem evidências visuais a extrair.")
            return defect_maps_output

        # Mapeia as âncoras e chama os métodos da sua classe CLIPSegModel
        for idx, (original_concept, prob) in enumerate(conceitos_eng.items()):
            
            visual_target = original_concept
            for key, val in self.visual_anchors.items():
                if key in original_concept.lower():
                    visual_target = val
                    break
            
            # Repassa a imagem, o prompt mapeado e a flag de cor dinâmica para o seu método funcional
            try:
                overlay_path = self.segmentador.generate_defect_overlay(
                    image_path=image_path, 
                    prompts=[visual_target], 
                    overlay_color=overlay_color, 
                    prompt_index=idx
                )
                
                # Tradução e estruturação dos metadados textuais do laudo
                concept_pt = self.concepts_map.get(original_concept, original_concept)
                defect_maps_output.append({
                    "conceito": concept_pt,
                    "probabilidade": prob,
                    "defect_map_path": overlay_path,
                    "prompt_usado": visual_target
                })
            except Exception as e:
                print(f"❌ Falha ao processar o overlay para '{visual_target}': {e}")

        return defect_maps_output