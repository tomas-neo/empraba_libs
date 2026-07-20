import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy import stats

def executar_analise_loadings():
    print("=" * 70)
    print("ANÁLISE DE LOADINGS (PESOS) - O RAIO-X DA PCA")
    print("=" * 70)

    # ---------------------------------------------------------
    # MENU INTERATIVO
    # ---------------------------------------------------------
    print("\nQual detector deseja investigar?")
    print("[1] VIS (Visível)")
    print("[2] UV (Ultravioleta)")
    
    escolha = input("\nDigite 1 ou 2 e aperte Enter: ").strip()
    if escolha == '1':
        detector_escolhido = "VIS"
    elif escolha == '2':
        detector_escolhido = "UV"
    else:
        print("\n[ERRO] Opção inválida.")
        return
        
    print(f"\n=> A extrair a matriz química do detector {detector_escolhido}...\n")

    pasta_resultados = Path("RESULTADOS_LIBS")
    
    arquivos_csv = [
        f for f in pasta_resultados.rglob("*_processado.csv") 
        if "backup" not in str(f).lower() 
        and any(detector_escolhido.lower() == pasta.lower() for pasta in f.parts)
    ]
    
    if not arquivos_csv:
        print(f"[ERRO] Nenhum ficheiro encontrado para {detector_escolhido}.")
        return

    matriz_X = []
    tamanho_padrao = None
    comprimentos_onda = None

    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        
        # Tenta extrair o eixo X (Comprimento de onda) do primeiro ficheiro
        if comprimentos_onda is None:
            # Procura por uma coluna que se chame algo como 'comprimento_onda' ou 'nm'
            col_onda = next((col for col in df.columns if 'onda' in col.lower() or 'wave' in col.lower() or 'nm' in col.lower()), None)
            if col_onda:
                comprimentos_onda = df[col_onda].values
            else:
                # Se não encontrar o nome exato, usa o número do ponto (pixel)
                comprimentos_onda = np.arange(len(df))
                
        espectro = df['intensidade_pre_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)

    matriz_X = np.array(matriz_X)
    
    # ---------------------------------------------------------
    # 2. FILTRAGEM DE OUTLIERS (Para garantir coerência com a PCA anterior)
    # ---------------------------------------------------------
    scaler_ini = StandardScaler()
    X_esc_ini = scaler_ini.fit_transform(matriz_X)
    pca_ini = PCA(n_components=2)
    X_pca_ini = pca_ini.fit_transform(X_esc_ini)
    z_scores = np.abs(stats.zscore(X_pca_ini))
    mascara_limpa = (z_scores < 3).all(axis=1)
    matriz_X_limpa = matriz_X[mascara_limpa]

    # ---------------------------------------------------------
    # 3. CÁLCULO DOS LOADINGS (A Mágica)
    # ---------------------------------------------------------
    scaler_final = StandardScaler()
    X_escalonado_final = scaler_final.fit_transform(matriz_X_limpa)

    pca_final = PCA(n_components=2)
    pca_final.fit(X_escalonado_final)
    
    var_pc1 = pca_final.explained_variance_ratio_[0] * 100
    var_pc2 = pca_final.explained_variance_ratio_[1] * 100

    # Extrai os pesos (Loadings) da PC1 e da PC2
    loadings_pc1 = pca_final.components_[0]
    loadings_pc2 = pca_final.components_[1]

    # ---------------------------------------------------------
    # 4. CONSTRUÇÃO DO GRÁFICO DUPLO
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Gráfico da PC1
    ax1.plot(comprimentos_onda, loadings_pc1, color='#2c3e50', linewidth=1.5)
    ax1.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_title(f'Loadings da PC1 ({var_pc1:.1f}% da variação) - O que mais separou os dados?', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peso no PC1', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Gráfico da PC2
    ax2.plot(comprimentos_onda, loadings_pc2, color='#27ae60', linewidth=1.5)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title(f'Loadings da PC2 ({var_pc2:.1f}% da variação) - A segunda maior influência', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Comprimento de Onda (nm) / Índice do Pixel', fontsize=10)
    ax2.set_ylabel('Peso no PC2', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'Raio-X da PCA: Gráfico de Loadings ({detector_escolhido})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print(f"\n[SUCESSO]")
    plt.show()

if __name__ == '__main__':
    executar_analise_loadings()
