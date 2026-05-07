import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
import sys

# --- RESOLVENDO CAMINHOS ABSOLUTOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(BASE_DIR)

try:
    from networks.custom_detector import MultimodalDetector
except ImportError:
    print(f"❌ Erro: Não foi possível encontrar 'networks.custom_detector' em {BASE_DIR}")
    sys.exit(1)

# --- CONFIGURAÇÕES DE DISPOSITIVO ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MAPEAMENTO DE DIRETÓRIOS ---
TEST_DIR = os.path.join(BASE_DIR, "data", "test_set", "test_set")
METADATA_CSV = os.path.join(BASE_DIR, "data", "test_set_metadata.csv")
WEIGHTS_ROOT = os.path.join(BASE_DIR, "weights")

# ARQUIVOS DE SAÍDA
OUTPUT_SCORES_CSV = os.path.join(SCRIPT_DIR, "resultados_parciais_slides.csv")
FINAL_METRICS_CSV = os.path.join(SCRIPT_DIR, "tabela_comparativa_slides.csv")

# --- DEFINIÇÃO DOS MODELOS DO DUELO ---
MODELOS_TESTE = {
    "Cozzolino_Baseline": os.path.join("clipdet_latent10k", "weights.pth"), 
    "Nossa_Proposta_Ep7": os.path.join("CLIP_Custom_Detector", "frankenstein_ep7.pth")
}

# Resolvendo erro de SSL no Windows 
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]

# --- FUNÇÃO DE MÉTRICAS ---
def compute_final_table(scores_df, metadata_path):
    metadata = pd.read_csv(metadata_path)
    table = metadata.merge(scores_df, on='filename')
    
    if 'typ' not in table.columns:
        print(f"❌ Coluna 'typ' não encontrada. Colunas: {table.columns}")
        return None
    
    list_typs = sorted([t for t in table['typ'].unique() if t != 'real'])
    table['label'] = table['typ'] != 'real'

    tab_metrics = pd.DataFrame(index=list(MODELOS_TESTE.keys()), columns=list_typs)
    
    for typ in list_typs:
        tab_typ = table[table['typ'].isin(['real', typ])]
        for alg in tab_metrics.index:
            if alg in table.columns:
                score = tab_typ[alg].values
                label = tab_typ['label'].values
                if len(np.unique(label)) > 1:
                    tab_metrics.loc[alg, typ] = metrics.roc_auc_score(label, score)
    
    tab_metrics['AVG'] = tab_metrics.mean(axis=1)
    return tab_metrics

def main():
    print(f"🔍 Buscando imagens em: {TEST_DIR}")
    all_images = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, f))
    
    if not all_images:
        print(f"❌ Nenhuma imagem encontrada em: {TEST_DIR}")
        return

    print(f"✅ {len(all_images)} imagens encontradas.")

    df_results = pd.DataFrame({"filename": [os.path.basename(p) for p in all_images]})
    model = MultimodalDetector().to(DEVICE)
    model.eval()

    for nome_exibicao, caminho_relativo in MODELOS_TESTE.items():
        ckpt_path = os.path.join(WEIGHTS_ROOT, caminho_relativo)
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ PULEI: {nome_exibicao} não encontrado em {ckpt_path}")
            continue
        
        print(f"\n🚀 Carregando e Avaliando: {nome_exibicao}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        
        # --- LÓGICA DE CARREGAMENTO ULTRA-ROBUSTA ---
        state_dict = None
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
            
        # strict=False ignora as chaves do CLIP que não estão no .pth
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ Pesos de {nome_exibicao} carregados (Modo Flexível).")
        # ---------------------------------------
        
        scores = []
        with torch.no_grad():
            for p in tqdm(all_images, desc=f"Progresso {nome_exibicao}"):
                try:
                    img = model.preprocess(Image.open(p).convert('RGB')).unsqueeze(0).to(DEVICE)
                    output = model(img)
                    score = torch.sigmoid(output).item()
                    scores.append(score)
                except:
                    scores.append(0.5)
        
        df_results[nome_exibicao] = scores

    df_results.to_csv(OUTPUT_SCORES_CSV, index=False)
    
    if os.path.exists(METADATA_CSV):
        print("\n Gerando tabela comparativa...")
        res_table = compute_final_table(df_results, METADATA_CSV)
        if res_table is not None:
            print("\n--- RESULTADOS FINAIS (AUC) ---")
            print(res_table.to_string(float_format=lambda x: '%5.3f' % x))
            res_table.to_csv(FINAL_METRICS_CSV)
            print(f"\n Concluído! Tabela salva em: {FINAL_METRICS_CSV}")
    else:
        print("\n Fim do loop. Metadata não encontrado, apenas scores salvos.")

if __name__ == "__main__":
    main()