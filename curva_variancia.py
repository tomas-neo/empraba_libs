import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def plotar_curva_variancia():
    print("=" * 70)
    print("GERADOR DE CURVA DE VARIÂNCIA EXPLICADA ACUMULADA (PCA)")
    print("=" * 70)

    # Menu Interativo
    print("\nQual detector você deseja analisar a curva?")
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

    pasta_resultados = Path("RESULTADOS_LIBS")
    
    # Busca Robusta com Blindagem
    arquivos_csv = [
        f for f in pasta_resultados.rglob("*_processado.csv") 
        if "backup" not in str(f).lower() 
        and any(detector_escolhido.lower() == pasta.lower() for pasta in f.parts)
    ]
    
    if not arquivos_csv:
        print(f"[ERRO] Nenhum arquivo processado encontrado para {detector_escolhido}.")
        return

    matriz_X = []
    tamanho_padrao = None

    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo)
        espectro = df['intensidade_pre_processada'].values
        
        if tamanho_padrao is None:
            tamanho_padrao = len(espectro)
            
        if len(espectro) != tamanho_padrao:
            continue
            
        matriz_X.append(espectro)

    matriz_X = np.array(matriz_X)
    print(f"Matriz validada! Analisando a variância de {len(matriz_X)} espectros.")

    # Matemática da Variância
    scaler = StandardScaler()
    X_escalonado = scaler.fit_transform(matriz_X)

    n_comp = min(20, len(matriz_X))
    pca = PCA(n_components=n_comp)
    pca.fit(X_escalonado)
    
    var_explicada = pca.explained_variance_ratio_ * 100
    var_acumulada = np.cumsum(var_explicada)

    # Desenhando o Gráfico
    plt.figure(figsize=(10, 6))
    indices = np.arange(1, n_comp + 1)
    plt.plot(indices, var_acumulada, marker='o', color='#3b2c85', linewidth=2, markersize=8)

    limite_95 = np.where(var_acumulada >= 95)[0]
    if len(limite_95) > 0:
        pc_alvo = limite_95[0] + 1
        val_alvo = var_acumulada[limite_95[0]]
        
        plt.axvline(x=pc_alvo, color='#cc3333', linestyle='--', linewidth=1.5)
        plt.axhline(y=val_alvo, color='#cc3333', linestyle='--', linewidth=1.5)
        
        plt.annotate(f'PCA{pc_alvo}\n({val_alvo:.1f}%)',
                     xy=(pc_alvo, val_alvo),
                     xytext=(pc_alvo - 3, val_alvo + 2.5),
                     arrowprops=dict(facecolor='#cc3333', edgecolor='#cc3333', shrink=0.05, width=2, headwidth=8),
                     color='#cc3333', fontsize=12, fontweight='bold', ha='center')

    plt.title(f'Variância Explicada Acumulada - PCA ({detector_escolhido})', fontsize=14, fontweight='bold')
    plt.xlabel('Índices dos componentes', fontsize=12)
    plt.ylabel('% Variância Explicada Acumulada', fontsize=12)
    
    plt.xticks(indices)
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.ylim(0, 105)
    plt.tight_layout()
    
    
    print(f"\n[SUCESSO]")
    plt.show()

if __name__ == '__main__':
    plotar_curva_variancia()
