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

    # Define a pasta onde estão os arquivos processados
    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # Busca todos os arquivos que terminam com '_processado.csv'
    arquivos_csv = list(pasta_resultados.rglob("*_processado.csv"))
    
    if not arquivos_csv:
        print("[ERRO] Nenhum arquivo processado encontrado. Verifique as pastas.")
        return

    print(f"Encontrados {len(arquivos_csv)} espectros para a matriz da PCA.")

    matriz_X = []
    labels_amostras = []

    # 1. Montagem da Matriz de Dados
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        
        # Extrai apenas a coluna da intensidade já pré-processada (área normalizada, sem baseline)
        espectro = df['intensidade_pre_processada'].values
        matriz_X.append(espectro)
        
        # 2. Extração Automática de Rótulos (Labels) a partir do caminho do arquivo
        caminho_texto = str(arquivo).lower()
        
        # Identifica a Marca
        if "maxgreen" in caminho_texto:
            marca = "MaxGreen"
        elif "plantfertil" in caminho_texto:
            marca = "PlantFertil"
        else:
            marca = "Desconhecida"
            
        # Identifica a Formulação NPK
        if "10-10-10" in caminho_texto:
            formula = "10-10-10"
        elif "4-14-8" in caminho_texto:
            formula = "4-14-8"
        else:
            formula = ""
            
        # Cria o rótulo final (Ex: "MaxGreen 10-10-10")
        rotulo_final = f"{marca} {formula}".strip()
        labels_amostras.append(rotulo_final)

    # Converte listas para arrays do NumPy (formato exigido pelo scikit-learn)
    matriz_X = np.array(matriz_X)
    labels_amostras = np.array(labels_amostras)

    # 3. Pré-processamento Quimiométrico (Padronização / Auto-scaling)
    # A padronização garante que picos muito altos não esmaguem picos menores de nutrientes importantes
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    # 4. Cálculo da PCA
    print("\nCalculando as Componentes Principais...")
    pca = PCA(n_components=2) # Queremos apenas PC1 e PC2 para o gráfico 2D
    X_pca = pca.fit_transform(X_escalonado)
    
    # Porcentagem de variância explicada por cada componente
    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100

    # 5. Construção do Gráfico de Escores (Scores Plot)
    plt.figure(figsize=(10, 7))
    
    # Criamos um DataFrame temporário só para facilitar o plot com o Seaborn
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Amostra': labels_amostras
    })

    # Paleta de cores contrastantes para separar bem os clusters
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Amostra', 
        palette='Set1', 
        data=df_pca, 
        s=100, 
        alpha=0.8, 
        edgecolor='black'
    )

    # Linhas de eixo cruzando no zero (Padrão Quimiométrico)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)

    plt.title('PCA - Gráfico de Escores (Fertilizantes NPK)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(title='Formulação / Marca', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Salva o gráfico na raiz
    nome_grafico = "PCA_Scores_Plot.png"
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico da PCA gerado e salvo como '{nome_grafico}'!")
    plt.show()

if __name__ == '__main__':
    executar_pca_libs()
