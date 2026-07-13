import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import seaborn as sns
from scipy import stats

def identificar_e_plotar_outliers():
    print("=" * 70)
    print("CAÇADOR DE OUTLIERS - ANÁLISE DE COMPONENTES PRINCIPAIS (PCA)")
    print("=" * 70)

    # Menu Interativo
    print("\nQual detector você deseja rastrear os outliers?")
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
        
    print(f"\n=> Rastreando os dados do detector {detector_escolhido}...\n")

    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # Busca Robusta com Blindagem
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
    nomes_arquivos = []
    tamanho_padrao = None

    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        espectro = df['intensidade_pre_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)
        nomes_arquivos.append(arquivo.name)
        
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
    
    print(f"Matriz validada! Analisando {len(matriz_X)} espectros em busca de anomalias.\n")

    # PCA Bruta e Cálculo de Outliers
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_escalonado)
    
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    z_scores = np.abs(stats.zscore(X_pca))
    outlier_mask = (z_scores >= 3).any(axis=1)
    
    # Dedurando no Terminal
    print("!" * 50)
    print("RELATÓRIO DE TIROS ANÔMALOS (OUTLIERS)")
    print("!" * 50)
    
    total_outliers = np.sum(outlier_mask)
    if total_outliers == 0:
        print("Nenhum outlier encontrado! Todos os dados estão coesos.")
    else:
        print(f"Foram encontrados {total_outliers} espectros anômalos:\n")
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                z_max = max(z_scores[i])
                print(f"-> Arquivo: {nomes_arquivos[i]}")
                print(f"   Marca: {labels_amostras[i]} | Distância Z-Score: {z_max:.2f}\n")

    # Construção do Gráfico
    plt.figure(figsize=(10, 7))
    status_pontos = ['Outlier' if out else 'Normal' for out in outlier_mask]
    
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Amostra': labels_amostras,
        'Status': status_pontos
    })

    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Amostra', 
        palette='Set1', 
        data=df_pca[df_pca['Status'] == 'Normal'], 
        s=100, 
        alpha=0.8, 
        edgecolor='black'
    )
    
    if total_outliers > 0:
        sns.scatterplot(
            x='PC1', y='PC2', 
            data=df_pca[df_pca['Status'] == 'Outlier'], 
            s=200, marker='X', color='red', edgecolor='black',
            label='Outliers Detectados'
        )

    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.title(f'PCA - Detecção de Outliers (Fertilizantes NPK) - Detector {detector_escolhido}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # ---------------------------------------------------------
    # CRIAÇÃO AUTOMÁTICA DA PASTA DE DESTINO
    # ---------------------------------------------------------
    pasta_destino = Path("PCA_Com_Outliers")
    pasta_destino.mkdir(exist_ok=True) # Cria a pasta se ela não existir
    
    nome_grafico = pasta_destino / f"PCA_Outliers_Destacados_{detector_escolhido}.png"
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico salvo em: {nome_grafico}")
    plt.show()

if __name__ == '__main__':
    identificar_e_plotar_outliers()
