import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from scipy import stats

def executar_pca_3d():
    print("=" * 70)
    print("GERADOR DE PCA 3D (PC1, PC2 e PC3)")
    print("=" * 70)

    # ---------------------------------------------------------
    # MENU INTERATIVO
    # ---------------------------------------------------------
    print("\nQual detector deseja visualizar em 3D?")
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
        
    print(f"\n=> A carregar dados do detector {detector_escolhido} para o espaço 3D...\n")

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
    labels_amostras = []
    tamanho_padrao = None

    # 1. Montagem da Matriz e Escudo Protetor
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        espectro = df['intensidade_pre_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)
        
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

    # 2. Filtragem de Outliers (Z-Score)
    scaler_ini = StandardScaler()
    X_esc_ini = scaler_ini.fit_transform(matriz_X)
    pca_ini = PCA(n_components=3) # Agora testamos em 3 dimensões
    X_pca_ini = pca_ini.fit_transform(X_esc_ini)
    
    z_scores = np.abs(stats.zscore(X_pca_ini))
    mascara_limpa = (z_scores < 3).all(axis=1)
    
    matriz_X_limpa = matriz_X[mascara_limpa]
    labels_limpos = labels_amostras[mascara_limpa]

    # 3. PCA Definitiva (Com 3 Componentes)
    scaler_final = StandardScaler()
    X_escalonado_final = scaler_final.fit_transform(matriz_X_limpa)

    pca_final = PCA(n_components=3)
    X_pca_final = pca_final.fit_transform(X_escalonado_final)
    
    var_pc1 = pca_final.explained_variance_ratio_[0] * 100
    var_pc2 = pca_final.explained_variance_ratio_[1] * 100
    var_pc3 = pca_final.explained_variance_ratio_[2] * 100

    # 4. Construção do Gráfico 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d') # Ativa o motor 3D do Matplotlib

    # Paleta de cores básica para garantir o contraste
    cores = {'PlantFertil 4-14-8': '#e74c3c', 
             'MaxGreen 4-14-8': '#3498db', 
             'PlantFertil 10-10-10': '#2ecc71', 
             'MaxGreen 10-10-10': '#9b59b6'}

    grupos_unicos = np.unique(labels_limpos)

    # Plotamos grupo por grupo para criar a legenda corretamente
    for grupo in grupos_unicos:
        # Encontra os índices de todos os pontos que pertencem a este grupo
        indices = np.where(labels_limpos == grupo)
        
        # Define a cor (se houver alguma amostra desconhecida, usa cinza)
        cor_grupo = cores.get(grupo, '#95a5a6')
        
        # Desenha os pontos no espaço 3D (X, Y, Z)
        ax.scatter(
            X_pca_final[indices, 0], # Eixo X (PC1)
            X_pca_final[indices, 1], # Eixo Y (PC2)
            X_pca_final[indices, 2], # Eixo Z (PC3)
            label=grupo, 
            s=80, 
            color=cor_grupo, 
            edgecolor='black', 
            alpha=0.8
        )

    # Configuração dos eixos
    ax.set_title(f'Gráfico de Escores 3D - PCA ({detector_escolhido})', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel(f'PC1 ({var_pc1:.1f}%)', fontsize=11, labelpad=10)
    ax.set_ylabel(f'PC2 ({var_pc2:.1f}%)', fontsize=11, labelpad=10)
    ax.set_zlabel(f'PC3 ({var_pc3:.1f}%)', fontsize=11, labelpad=10)

    # Ajusta o ângulo de visão inicial (elevação, azimute)
    ax.view_init(elev=20, azim=45) 
    
    ax.legend(title='Formulação / Marca', bbox_to_anchor=(1.1, 0.9), loc='upper left')

    
    print(f"\n[SUCESSO]")
    print("DICA: Quando a janela do gráfico abrir, clique e arraste com o rato para rodar o cubo em 360º!\n")
    
    plt.show()

if __name__ == '__main__':
    executar_pca_3d()
