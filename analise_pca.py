import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns

def executar_pca_libs():
    print("=" * 70)
    print("INICIANDO ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("=" * 70)

    # ---------------------------------------------------------
    # MENU INTERATIVO PARA ESCOLHA DO DETECTOR
    # ---------------------------------------------------------
    print("\nQual detector você deseja usar para construir esta PCA?")
    print("[1] VIS (Visível)")
    print("[2] UV (Ultravioleta)")
    
    escolha = input("\nDigite 1 ou 2 e aperte Enter: ").strip()
    
    if escolha == '1':
        detector_escolhido = "VIS"
    elif escolha == '2':
        detector_escolhido = "UV"
    else:
        print("\n[ERRO] Opção inválida. Execute o script novamente e digite apenas 1 ou 2.")
        return
        
    print(f"\n=> Excelente! Filtrando apenas os dados do detector {detector_escolhido}...\n")
    # ---------------------------------------------------------

    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # BUSCA ROBUSTA: Verifica se a pasta se chama EXATAMENTE "uv" ou "vis" 
    arquivos_csv = [
        f for f in pasta_resultados.rglob("*_processado.csv") 
        if "backup" not in str(f).lower() 
        and any(detector_escolhido.lower() == pasta.lower() for pasta in f.parts)
    ]
    
    if not arquivos_csv:
        print(f"[ERRO] Nenhum arquivo processado encontrado para o detector {detector_escolhido}.")
        return

    print(f"Encontrados {len(arquivos_csv)} arquivos potenciais para o detector {detector_escolhido}.")

    matriz_X = []
    labels_amostras = []
    tamanho_padrao = None # Variável escudo para travar o tamanho

    # 1. Montagem da Matriz de Dados (Com Proteção)
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        espectro = df['intensidade_pre_processada'].values
        
        # Define a régua de tamanho baseada no primeiro arquivo que entrar
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        # O ESCUDO: Se o tamanho for diferente, ignora silenciosamente o arquivo impostor
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)
        
        # 2. Extração Automática de Rótulos (Labels)
        caminho_texto = str(arquivo).lower()
        
        if "maxgreen" in caminho_texto:
            marca = "MaxGreen"
        elif "plantfertil" in caminho_texto:
            marca = "PlantFertil"
        else:
            marca = "Desconhecida"
            
        if "10-10-10" in caminho_texto:
            formula = "10-10-10"
        elif "4-14-8" in caminho_texto:
            formula = "4-14-8"
        else:
            formula = ""
            
        rotulo_final = f"{marca} {formula}".strip()
        labels_amostras.append(rotulo_final)

    # Converte listas para arrays
    matriz_X = np.array(matriz_X)
    labels_amostras = np.array(labels_amostras)
    
    print(f"Matriz validada com sucesso! Analisando {len(matriz_X)} espectros homogêneos.")

    # 3. Pré-processamento Quimiométrico
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    # 4. Cálculo da PCA
    print("Calculando as Componentes Principais...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_escalonado)
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 5. Construção do Gráfico
    plt.figure(figsize=(10, 7))
    
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Amostra': labels_amostras
    })

    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Amostra', 
        palette='Set1', 
        data=df_pca, 
        s=100, 
        alpha=0.8, 
        edgecolor='black'
    )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.title(f'PCA - Gráfico de Escores (Fertilizantes NPK) - Detector {detector_escolhido}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(title='Formulação / Marca', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    nome_grafico = f"PCA_Scores_Plot_{detector_escolhido}.png"
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico da PCA gerado e salvo como '{nome_grafico}'!")
    plt.show()

if __name__ == '__main__':
    executar_pca_libs()
