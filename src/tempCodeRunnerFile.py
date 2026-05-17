import os
from datasets import load_dataset
from torch.utils.data import DataLoader

def iniciar_dataset():
    print("🚀 Conectando ao Hugging Face para carregar o SID_Set...")
    
    cache_dir = None 

    # Carrega o dataset (Train e Validation) do Hugging Face
    # O download é inteligente: se cair, ele continua de onde parou.
    dataset = load_dataset("saberzl/SID_Set", cache_dir=cache_dir)
    
    print("\n📊 Dataset carregado com sucesso no Windows!")
    print(dataset)
    
    # Filtrando para manter apenas Imagens Reais (0) e Totalmente Sintéticas (1)
    # Ignorando por enquanto as adulteradas/tampered (2) para o seu classificador binário
    print("\n⚡ Filtrando o conjunto de treino (Apenas Real [0] e Full Synthetic [1])...")
    train_filtrado = dataset['train'].filter(lambda x: x['label'] in [0, 1])
    val_filtrado = dataset['validation'].filter(lambda x: x['label'] in [0, 1])
    
    print(f" -> Total Treino Filtrado: {len(train_filtrado)} imagens.")
    print(f" -> Total Validação Filtrada: {len(val_filtrado)} imagens.")
    
    # Inspecionando o primeiro elemento
    amostra = train_filtrado[0]
    print(f"\n🔍 Amostra de Teste Local:")
    print(f" - ID da Imagem: {amostra['img_id']}")
    print(f" - Rótulo (Label): {amostra['label']}")
    print(f" - Formato da Imagem: {amostra['image'].size} {amostra['image'].mode}")
    
    return train_filtrado, val_filtrado

if __name__ == '__main__':

    print('("🚀 Iniciando o processo de download e preparação do dataset no Windows 11...")')
    # No Windows 11, todo o ponto de partida de processamento de dados precisa estar aqui dentro
    train_data, val_data = iniciar_dataset()
    print("\n🎯 Tudo pronto para alimentar o CLIP e extrair os atributos de textura (LBP)!")