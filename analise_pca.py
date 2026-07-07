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

    # Define a pasta onde estão os arquivos processados
    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # Busca os arquivos ignorando o backup e filtrando exatamente pela escolha do usuário
    arquivos_csv = [f for f in pasta_resultados.rglob("*_processado.csv") if "backup" not in f.parts and detector_escolhido in f.parts]
    
    if not arquivos_csv:
        print(f"[ERRO] Nenhum arquivo processado encontrado para o detector {detector_escolhido}. Verifique as pastas.")
        return

    print(f"Encontrados {len(arquivos_csv)} espectros para a matriz da PCA ({detector_escolhido}).")

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
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    # 4. Cálculo da PCA
    print("Calculando as Componentes Principais...")
    pca = PCA(n_components=2)
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

    # Título dinâmico que muda dependendo se é UV ou VIS
    plt.title(f'PCA - Gráfico de Escores (Fertilizantes NPK) - Detector {detector_escolhido}', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({var_pc1:.1f}% da variância)', fontsize=12)
    plt.ylabel(f'PC2 ({var_pc2:.1f}% da variância)', fontsize=12)
    plt.legend(title='Formulação / Marca', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    # Salva o gráfico com um nome dinâmico para não subscrever arquivos
    nome_grafico = f"PCA_Scores_Plot_{detector_escolhido}.png"
    plt.savefig(nome_grafico, dpi=300, bbox_inches='tight')
    print(f"\n[SUCESSO] Gráfico da PCA gerado e salvo como '{nome_grafico}'!")
    plt.show()

if __name__ == '__main__':
    executar_pca_libs()
