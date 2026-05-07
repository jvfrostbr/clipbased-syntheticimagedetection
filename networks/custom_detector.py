import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import numpy as np
from skimage.feature import local_binary_pattern

class MultimodalDetector(nn.Module):
    def __init__(self, num_classes=1, prompt_length=16, lbp_radius=1, lbp_points=8):
        super(MultimodalDetector, self).__init__()
        
        # Definição das classes para o Prompt Tuning
        self.class_names = ["real image", "an AI-generated image"]
        
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
        
        # Parâmetros Aprendíveis (Soft Prompts)
        self.prompt_length = prompt_length
        self.soft_prompts = nn.Parameter(torch.randn(prompt_length, 768) * 0.02)
        
        # MLP de classificação final
        img_dim = self.clip_model.visual.output_dim    
        txt_dim = self.clip_model.text_projection.shape[1] 
        
        
        # Entrada: CLIP Imagem + CLIP Texto (Tuned) + LBP
        input_mlp_dim = img_dim + (txt_dim * 2) + self.lbp_dim 
        
        self.mlp = nn.Sequential(
            nn.Linear(input_mlp_dim, 512),
            nn.BatchNorm1d(512), # normalização para equilibrar as dimensões CLIP vs LBP
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def extract_lbp_features(self, x):
        """ Extrai histogramas de textura LBP (Processamento em Batch) """
        device = x.device
        
        # Conversão manual RGB para Tons de Cinza vi
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

    def get_text_features(self, label_text, device):
        """ Lógica do 'Sanduíche' de Prompt Tuning com suporte a múltiplos tokens """
        
        # Tokenizando a classe ("an AI-generated image" e "real image") para obter os embeddings originais
        tokens = open_clip.tokenize([label_text]).to(device) 
        
        # Conversão para embeddings usando a camada de tokenização do CLIP 
        with torch.no_grad():
            embedding_real = self.clip_model.token_embedding(tokens) 
            
        # Identificação dinâmica do tamanho da classe (ex: "an AI-generated image")
        eos_index_orig = tokens.argmax(dim=-1).item()
        
        # Isolando os embeddings (SOS, Nome da Classe, e o resto)
        prefix = embedding_real[:, :1, :] # 
        class_content = embedding_real[:, 1:eos_index_orig, :] 
        num_class_tokens = class_content.shape[1]
        
        # Calculando quanto de padding/sufixo sobra para manter 77 tokens no total
        suffix_len = 77 - 1 - self.prompt_length - num_class_tokens
        suffix = embedding_real[:, eos_index_orig : eos_index_orig + suffix_len, :]
        
        # Montando o novo embedding com os Soft Prompts inseridos entre o SOS e o conteúdo da classe
        # SOS + soft_prompts + classe + EOS/ZEROS
        tuned_embedding = torch.cat([
            prefix, 
            self.soft_prompts.unsqueeze(0), 
            class_content, 
            suffix
        ], dim=1)
        
        # Processamento no Transformer do CLIP
        x = tuned_embedding + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)
        
        #  Extração no novo índice do EOS 
        new_eos_index = eos_index_orig + self.prompt_length
        text_features = x[torch.arange(x.shape[0]), new_eos_index] @ self.clip_model.text_projection
        
        return text_features


    def forward(self, images):
        batch_size = images.shape[0]
        device = images.device
        
        # Extração do vetor de características da imagem usando o CLIP
        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
        
        # Extraindo o conhecimento do prompt tuning para AMBOS os conceitos
        text_feat_real = self.get_text_features(self.class_names[0], device) # "real image"
        text_feat_fake = self.get_text_features(self.class_names[1], device) # "AI-generated"
        
        # Normalizamos (boa prática para manter as escalas)
        text_feat_real = F.normalize(text_feat_real, dim=-1).repeat(batch_size, 1)
        text_feat_fake = F.normalize(text_feat_fake, dim=-1).repeat(batch_size, 1)
        
        # Extração de Textura via LBP
        lbp_features = self.extract_lbp_features(images)
        
       # Fusão e Veredito Final
        combined = torch.cat((image_features, text_feat_real, text_feat_fake, lbp_features), dim=1)
        
        return self.mlp(combined)