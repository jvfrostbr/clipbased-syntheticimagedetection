import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Resolvendo o erro chato pra kct de SSL no Windows 
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
 
# Ajustando o caminho para importar o modelo customizado de networks.custom_detector
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    from networks.custom_detector import MultimodalDetector
except ImportError:
    print(" Erro: Não foi possível encontrar 'networks.custom_detector'.")
    sys.exit(1)

class FrankensteinFolderDataset(Dataset):
    def __init__(self, root_dir, preprocess):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.samples = []
        self.class_to_idx = {'real': 0, 'fake': 1}

        # Carregando os caminhos das imagens e seus rótulos
        for class_name in ['real', 'fake']:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for f in files:
                    self.samples.append((os.path.join(class_dir, f), self.class_to_idx[class_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"Nenhuma imagem encontrada em {root_dir}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            return self.preprocess(image), label
        except:
            return self.__getitem__((idx + 1) % len(self.samples))

def main():
    # Configurações de Treino
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATH_RAIZ = os.path.join(BASE_DIR, 'data', 'train_set', 'dataset_frankenstein')
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights', 'CLIP_Custom_Detector')
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 1e-4
    START_EPOCH = 0

    print(f"🚀 Iniciando motor em: {DEVICE}")

    # Inicializa o detector
    model = MultimodalDetector().to(DEVICE)

    # Otimizador com configuração diferenciada para soft prompts e MLP
    optimizer = optim.AdamW([
        {'params': [model.soft_prompts], 'lr': LR},
        {'params': model.mlp.parameters(), 'lr': LR}
    ], weight_decay=0.01)

    # Definindo a função de perda 
    criterion = nn.BCEWithLogitsLoss()

    # Verificação de checkpoints para retomar o treino de onde parou
    checkpoints = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pth')]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('ep')[-1].split('.')[0]))[-1]
        ckpt_path = os.path.join(WEIGHTS_DIR, last_checkpoint)
        
        print(f" Checkpoint encontrado: {last_checkpoint}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            START_EPOCH = checkpoint['epoch']
            print(f"⏩ Continuando da Época {START_EPOCH + 1}")
        except RuntimeError as e:
            print(f"⚠️ Erro de arquitetura: {e}")
            print("Arquitetura atual não bate com o checkpoint. Começando do zero.")
            START_EPOCH = 0

    # Verificação de conclusão prévia do treino
    if START_EPOCH >= EPOCHS:
        print(f"✅ Treino já concluído (Época {START_EPOCH}). Se quiser treinar mais, aumente o valor de EPOCHS.")
        return

    # Carregando o dataset 
    dataset = FrankensteinFolderDataset(root_dir=PATH_RAIZ, preprocess=model.preprocess)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Loop de Treino
    model.train()
    for epoch in range(START_EPOCH, EPOCHS):
        
        # Variáveis para monitoramento de perda e acurácia
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Época [{epoch+1}/{EPOCHS}]")
        
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            
            # Treino dos soft prompts e MLP
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Atualização de métricas
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=(running_loss/len(train_loader)), acc=(correct/total))

        # salvando o checkpoint ao final de cada época
        save_path = os.path.join(WEIGHTS_DIR, f'frankenstein_ep{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(), 
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': (correct / total)
        }, save_path)

    print("\n✨ Treino concluído com sucesso!")

if __name__ == '__main__':
    main()