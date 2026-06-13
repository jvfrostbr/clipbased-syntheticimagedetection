import os
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image
from lang_sam import LangSAM

class ForensicExplainer:
    def __init__(self, clip_base_model, preprocess_fn, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inicializando Módulo de Explicabilidade Forense no {self.device}...")

        # Carregando o CLIP Vit-L/14 para extração de conceitos
        self.clip_model = clip_base_model
        self.preprocess = preprocess_fn

        #  Carregando os conceitos e âncoras definidos nos arquivos .txt
        self._load_configurations()

        # Carregando o LangSAM para segmentação
        print("Carregando o LangSAM, para segmentação de alta precisão")
        try:
            self.lang_sam = LangSAM() 
        except Exception as e:
            print(f"❌ Erro ao inicializar LangSAM: {e}")
            raise e

    def _load_configurations(self):
        """Lê os arquivos txt de conceitos e âncoras para memória."""
        self.concepts_eng = []      
        self.concepts_map = {}      
        self.visual_anchors = {}    

        # procura os arquivos de configuração na pasta config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config") 

        # Carrega os conceitos (Inglês -> Português)
        # e as âncoras visuais (termos-chave -> alvos de segmentação)
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
        """
            Extrai conceitos forenses da imagem usando CLIP, com limiar dinâmico.
        """
        conceitos_completos = self.concepts_eng + ["a high quality natural photograph"]

        try:
            # Preprocessamento oficial do CLIP para imagem
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenização oficial via OpenCLIP
            text_tokens = open_clip.tokenize(conceitos_completos).to(self.device)

            # Ajuste dinâmico do limiar para a probabilidade 
            # dos conceitos, baseado na classificação preliminar
            threshold = 0.25 if classificacao_preliminar in ["a real photograph", 0] else 0.10

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_tokens)

                # Normalização L2 para garantir comparabilidade entre as features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Produto Escalar + Softmax para obter as probabilidades dos conceitos
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = text_probs.cpu().numpy()[0]

            # Aplicando o threshold e mapeando os conceitos que passaram para o resultado final
            resultado = {}
            for i in range(len(self.concepts_eng)):
                if probs[i] > threshold: 
                    resultado[self.concepts_eng[i]] = float(probs[i])
            
            # Retorna do conceito mais provável para o menos provável
            return dict(sorted(resultado.items(), key=lambda item: item[1], reverse=True))

        except Exception as e:
            print(f"⚠️ Erro na análise de conceitos: {e}")
            return {}

    def explain_decision(self, image_path, label_eng, overlay_color="red"):
        """
        Extrai os conceitos / defeitos detectados, mapeia âncoras e aciona o LangSAM.
        """
        image_pil = Image.open(image_path).convert("RGB")
        
        # inicia a etapa de análise de conceitos / defeitos forenses
        conceitos_eng = self.analisar_conceitos(image_path, classificacao_preliminar=label_eng)
        
        defect_maps_output = []
        os.makedirs("outputs/defect_maps", exist_ok=True)

        if not conceitos_eng:
            print("⚠️ Nenhum conceito superou o threshold. Sem evidências visuais a extrair.")
            return defect_maps_output

        # inicio do mapeamento visual dos conceitos para os alvos de segmentação do LangSAM
        for idx, (original_concept, prob) in enumerate(conceitos_eng.items()):
            
            visual_target = original_concept
            for key, val in self.visual_anchors.items():
                if key in original_concept.lower():
                    visual_target = val
                    break
            
            print(f"LangSAM Alvo: '{visual_target}' (Origem: {original_concept})")
            
            # Inferência LangSAM (Grounding DINO + SAM) 
            # para obter as máscaras de segmentação correspondentes ao conceito detectado
            masks, boxes, phrases, logits = self.lang_sam.predict(image_pil, visual_target)
            
            if len(masks) == 0:
                print(f"       [-] Nenhuma região encontrada para '{visual_target}'.")
                continue

            # Unifica todas as instâncias encontradas na imagem em uma máscara única
            unified_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.float32)
            
            # Inicio da Geração do Overlay Colorido, baseado na máscara do LangSAM
            img_np = np.array(image_pil)
            color_layer = np.zeros_like(img_np)

            # Isolando os pixels da imagem real que estão sob a máscara
            pixels_afetados = img_np[unified_mask > 0]
            
            # Calculando a cor RGB média dessa região da mascara
            # para orientar a escolha da cor de destaque
            if len(pixels_afetados) > 0:
                cor_media = pixels_afetados.mean(axis=0)
            else:
                cor_media = np.array([0, 0, 0])
                
            # Definindo as possíveis cores para a mascara
            paleta_neon = [
                np.array([255, 0, 0]),    # Vermelho
                np.array([0, 255, 0]),    # Verde
                np.array([0, 0, 255]),    # Azul
                np.array([255, 255, 0]),  # Amarelo
                np.array([255, 0, 255]),  # Magenta
                np.array([0, 255, 255])   # Ciano
            ]
            
            # Escolhendo a cor da paleta matematicamente mais distante da cor
            # da região da mascara para maior contraste e destaque visual
            melhor_cor = max(paleta_neon, key=lambda cor: np.linalg.norm(cor_media - cor))
            
            # Aplicando a cor vencedora em toda a região da mascara
            color_layer[:, :] = melhor_cor
            
            img_float = img_np.astype(np.float32) / 255.0
            color_float = color_layer.astype(np.float32) / 255.00
            
            # Expande a máscara para 3 canais (RGB)
            alpha = unified_mask[:, :, None]  
            
            # Fusão exata baseada na matriz pura do SAM
            overlay = (color_float * alpha * 0.5) + (img_float * (1.0 - (alpha * 0.2)))
            overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
            
            # Salvamento e registro físico
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            base = os.path.basename(image_path)
            name_without_ext = os.path.splitext(base)[0]
            
            overlay_path = f"outputs/defect_maps/{name_without_ext}_concept_{idx}.png"
            cv2.imwrite(overlay_path, overlay_bgr)
            
            # Tradução para o laudo final
            concept_pt = self.concepts_map.get(original_concept, original_concept)
            defect_maps_output.append({
                "conceito": concept_pt,
                "probabilidade": prob,
                "defect_map_path": overlay_path,
                "prompt_usado": visual_target
            })

        return defect_maps_output