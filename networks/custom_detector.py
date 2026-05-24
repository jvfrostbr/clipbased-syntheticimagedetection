import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import numpy as np
from skimage.feature import local_binary_pattern

class MultimodalDetector(nn.Module):
    def __init__(self, prompt_length=16, lbp_radius=1, lbp_points=8, use_prompt=True, use_lbp=True, multiclass=False):
        super(MultimodalDetector, self).__init__()
        
        self.use_prompt = use_prompt
        self.use_lbp = use_lbp
        self.multiclass = multiclass
        self.prompt_length = prompt_length
        
        # Configuração dinâmica baseada no escopo do experimento (Binário vs Multiclasse)
        if not self.multiclass:
            self.class_names = ["real image", "an AI-generated image"]
            num_prompt_classes = 2
            self.output_dim = 1  # Saída binária
        else:
            self.class_names = ["real image", "an AI-generated image", "a tampered manipulated image"]
            num_prompt_classes = 3
            self.output_dim = 3  # Saída multiclasse
        
        # Carregamento do CLIP ViT-L/14 (Com Pesos Congelados)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='openai'
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        # Configuração do LBP 
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.lbp_dim = lbp_points + 2 
        
        # Parâmetros Aprendíveis (Soft Prompts dedicados por classe)
        if self.use_prompt:
            self.soft_prompts = nn.Parameter(torch.randn(num_prompt_classes, prompt_length, 768) * 0.02)
        else:
            self.register_parameter('soft_prompts', None)
        
        # Cálculo dinâmico da entrada da MLP final
        img_dim = self.clip_model.visual.output_dim    
        input_mlp_dim = img_dim
        
        if self.use_prompt:
            txt_dim = self.clip_model.text_projection.shape[1]
            input_mlp_dim += (txt_dim * num_prompt_classes)
            
        if self.use_lbp:
            input_mlp_dim += self.lbp_dim 
        
        # Construção da MLP adaptada para o tamanho da variação
        self.mlp = nn.Sequential(
            nn.Linear(input_mlp_dim, 512),
            nn.BatchNorm1d(512), # Normalizando para equilibrar a escala das dimensões e escalas das features 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

    def extract_lbp_features(self, x):
        """ Extrai histogramas de textura LBP (Processamento em Batch) """
        device = x.device
        
        # Conversão manual RGB para Tons de Cinza
        with torch.no_grad():
            gray_imgs = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            gray_imgs = (gray_imgs * 255).cpu().numpy().astype(np.uint8)
        
        lbp_histograms = []
        for img in gray_imgs:
            lbp = local_binary_pattern(img, self.lbp_points, self.lbp_radius, method='uniform')
            
            # Criação do histograma normalizado do LBP
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_dim + 1), range=(0, self.lbp_dim))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            lbp_histograms.append(hist)
            
        return torch.tensor(np.array(lbp_histograms), dtype=torch.float32).to(device)

    def get_text_features(self, label_text, class_idx, device):
        """
            Extrai as características textuais do CLIP. Caso 'use_prompt' seja True, 
            realiza a injeção de Soft Prompts aprendíveis (Prompt Tuning) no espaço de 
            embeddings textuais de forma indexada por classe. Caso contrário, realiza 
            a extração nativa do CLIP Text Encoder.
            
            Para o fluxo com Prompt Tuning, a estrutura do vetor de tokens gerada segue 
            o seguinte alinhamento de componentes:
            [SOS] + [Soft Prompts Aprendíveis (16)] + [Conteúdo Semântico da Classe] + [EOS + Padding]
            
            A técnica expande o contexto textual com representações latentes otimizadas 
            durante o treino para guiar o alinhamento visual-textual forense.
        """
        tokens = open_clip.tokenize([label_text]).to(device) 
        
        # Sem prompt tunning: Extração direta e limpa do CLIP Text Encoder
        if not self.use_prompt:
            with torch.no_grad():
                text_features = self.clip_model.encode_text(tokens)
                return F.normalize(text_features, dim=-1)

        # Com prompt tunning: Injeção de Soft Prompts aprendíveis
        with torch.no_grad():
            embedding_real = self.clip_model.token_embedding(tokens) 
            
        eos_index_orig = tokens.argmax(dim=-1).item()
        
        # Isolando os componentes estruturais do texto original
        prefix = embedding_real[:, :1, :] 
        class_content = embedding_real[:, 1:eos_index_orig, :] 
        num_class_tokens = class_content.shape[1]
        
        suffix_len = 77 - 1 - self.prompt_length - num_class_tokens
        suffix = embedding_real[:, eos_index_orig : eos_index_orig + suffix_len, :]
        
        # Seleciona cirurgicamente a embedding aprendível da respectiva classe
        current_soft_prompt = self.soft_prompts[class_idx].unsqueeze(0)
        
        # Injeção dos soft prompts aprendíveis entre o prefixo (SOS) e o conteúdo
        tuned_embedding = torch.cat([
            prefix, 
            current_soft_prompt, 
            class_content, 
            suffix
        ], dim=1)
        
        # Passada manual pelo Transformer de Texto do CLIP
        x = tuned_embedding + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)
        
        new_eos_index = eos_index_orig + self.prompt_length
        text_features = x[torch.arange(x.shape[0]), new_eos_index] @ self.clip_model.text_projection
        
        return text_features

    def forward(self, images):
        batch_size = images.shape[0]
        device = images.device
        
        # Feature de Imagem (CLIP Visual Congelado)
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        features_compostas = [image_features]
        
        # Inclusão do Prompt Tuning de Texto
        if self.use_prompt:
            for idx, class_name in enumerate(self.class_names):
                text_feat = self.get_text_features(class_name, idx, device)
                text_feat = F.normalize(text_feat, dim=-1).repeat(batch_size, 1)
                features_compostas.append(text_feat)
            
        # Inclusão do Atributo de Baixo Nível (LBP)
        if self.use_lbp:
            lbp_features = self.extract_lbp_features(images)
            features_compostas.append(lbp_features)
            
        # Fusão por concatenação e veredito final pela MLP configurada
        combined = torch.cat(features_compostas, dim=1)
        return self.mlp(combined)