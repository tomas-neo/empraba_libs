import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns
from scipy import stats # <-- Nova ferramenta para detectar os outliers matematicamente

def executar_pca_libs():
    print("=" * 70)
    print("INICIANDO ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) COM FILTRO")
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
        
    print(f"\n=> Filtrando apenas os dados do detector {detector_escolhido}...\n")

    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # BUSCA ROBUSTA COM BLINDAGEM
    arquivos_csv = [
        f for f in pasta_resultados.rglob("*_processado.csv") 
        if "backup" not in str(f).lower() 
        and any(detector_escolhido.lower() == pasta.lower() for pasta in f.parts)
    ]
    
    if not arquivos_csv:
        print(f"[ERRO] Nenhum arquivo processado encontrado para o detector {detector_escolhido}.")
        return

    matriz_X = []
    labels_amostras = []
    tamanho_padrao = None

    # 1. Montagem da Matriz de Dados
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        espectro = df['intensidade_pre_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)
        
        # Extração de Rótulos (Labels)
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

    matriz_X = np.array(matriz_X)
    labels_amostras = np.array(labels_amostras)
    
    print(f"Matriz validada! Analisando {len(matriz_X)} espectros iniciais.")

    # ---------------------------------------------------------
    # 2. FILTRAGEM DE OUTLIERS (A Mágica Nova)
    # ---------------------------------------------------------
    # Fazemos uma PCA preliminar apenas para descobrir quem está fora da curva
    scaler_ini = StandardScaler()
    X_esc_ini = scaler_ini.fit_transform(matriz_X)
    pca_ini = PCA(n_components=2)
    X_pca_ini = pca_ini.fit_transform(X_esc_ini)

    # Calcula a distância (Z-score) de cada ponto
    z_scores = np.abs(stats.zscore(X_pca_ini))
    
    # Mantém apenas os espectros que estão a menos de 3 desvios padrões do centro
    # (Ou seja, expulsa os outliers absurdos)
    mascara_limpa = (z_scores < 3).all(axis=1)

    matriz_X_limpa = matriz_X[mascara_limpa]
    labels_amostras_limpos = labels_amostras[mascara_limpa]
    
    outliers_removidos = len(matriz_X) - len(matriz_X_limpa)
    print(f"-> FAXINA CONCLUÍDA: {outliers_removidos} 'tiros ruins' (outliers) foram expulsos da matriz!")
    print(f"-> Sobraram {len(matriz_X_limpa)} espectros confiáveis para o gráfico final.\n")

    # ---------------------------------------------------------
    # 3. PCA DEFINITIVA (Com os dados limpos)
    # ---------------------------------------------------------
    scaler_final = StandardScaler()
    X_escalonado_final = scaler_final.fit_transform(matriz_X_limpa)

    print("Calculando as Componentes Principais Definitivas...")
    pca_final = PCA(n_components=2)
    X_pca_final = pca_final.fit_transform(X_escalonado_final)
    
    var_pc1 = pca_final.explained_variance_ratio_[0] * 100
    var_pc2 = pca_final.explained_variance_ratio_[1] * 100

    # 4. Construção do Gráfico
    plt.figure(figsize=(10, 7))
    
    df_pca = pd.DataFrame({
        'PC1': X_pca_final[:, 0],
        'PC2': X_pca_final[:, 1],
        'Amostra': labels_amostras_limpos
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

    plt.title(f'PCA - Gráfico de Escores Filtrado (Fertilizantes NPK) - Detector {detector_escolhido}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(title='Formulação / Marca', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    print(f"[SUCESSO]!")
    plt.show()

if __name__ == '__main__':
    executar_pca_libs()
