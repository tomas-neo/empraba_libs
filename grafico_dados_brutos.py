import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# FUNÇÃO DE LEITURA ADAPTATIVA (Mantida para garantir compatibilidade)
# ==============================================================================
def carregar_arquivo_esf(caminho_arquivo):
    encodings = ['utf-8', 'utf-16', 'latin-1']
    
    for enc in encodings:
        try:
            with open(caminho_arquivo, 'r', encoding=enc, errors='ignore') as f:
                linhas = f.readlines()
            
            if not linhas:
                continue
                
            dados_num = []
            for linha in linhas:
                linha_txt = linha.strip()
                if not linha_txt:
                    continue
                
                if '\t' in linha_txt: partes = linha_txt.split('\t')
                elif ';' in linha_txt: partes = linha_txt.split(';')
                else: partes = linha_txt.split()
                
                if len(partes) >= 2:
                    try:
                        p_limpos = [p.strip().replace(',', '.') for p in partes]
                        valores = [float(p) for p in p_limpos if p]
                        
                        if len(valores) >= 3:
                            if valores[0] < 25000 and 180 <= valores[1] <= 1100:
                                dados_num.append([valores[1], valores[2]]) 
                            else:
                                dados_num.append([valores[0], valores[1]])
                        elif len(valores) == 2:
                            dados_num.append([valores[0], valores[1]])
                    except ValueError:
                        continue 
            
            if len(dados_num) > 0:
                dados_num = np.array(dados_num)
                dados_num = dados_num[dados_num[:, 0].argsort()]
                return dados_num[:, 0], dados_num[:, 1]
                
        except Exception:
            continue
            
    raise ValueError("Não foi possível extrair colunas numéricas válidas.")

# ==============================================================================
# PIPELINE DE PLOTAGEM BRUTA
# ==============================================================================
def gerar_graficos_brutos():
    print("=" * 60)
    print("GERADOR DE GRÁFICOS LIBS - DADOS BRUTOS")
    print("=" * 60)
    
    # Nome da pasta onde os gráficos serão salvos
    dir_saida = "graficos_brutos"
    os.makedirs(dir_saida, exist_ok=True)
    
    arquivos_esf = glob.glob("*.esf")
    
    if not arquivos_esf:
        print("[AVISO] Nenhum arquivo .esf encontrado nesta pasta.")
        return

    print(f"Encontrado(s) {len(arquivos_esf)} arquivo(s) para plotar.\n")
    
    for caminho_arquivo in arquivos_esf:
        nome_arquivo = os.path.basename(caminho_arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]
        
        try:
            # 1. Carrega os dados
            wl, intens_bruta = carregar_arquivo_esf(caminho_arquivo)
            
            # 2. Configura a figura
            plt.figure(figsize=(12, 6))
            
            # 3. Plota o dado puro
            plt.plot(wl, intens_bruta, color='#2c3e50', linewidth=1)
            
            # 4. Estilização do gráfico
            plt.title(f"Espectro Bruto: {nome_arquivo}", fontsize=12, fontweight='bold')
            plt.xlabel("Comprimento de Onda (nm)", fontsize=10)
            plt.ylabel("Intensidade Absoluta", fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.7)
            
            # Adiciona uma linha zerada suave para referência (ajuda a ver o fundo térmico)
            plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
            
            # 5. Salva a imagem
            caminho_salvar = os.path.join(dir_saida, f"{nome_base}_bruto.png")
            plt.tight_layout()
            plt.savefig(caminho_salvar, dpi=150)
            plt.close() # Importante para não consumir toda a memória RAM
            
            print(f"  [OK] Gráfico gerado: {nome_base}_bruto.png")
            
        except Exception as e:
            print(f"  [ERRO] Falha ao plotar {nome_arquivo}. Motivo: {str(e)}")

    print("\n" + "=" * 60)
    print(f"PROCESSO CONCLUÍDO! Verifique a pasta '{dir_saida}'.")
    print("=" * 60)

if __name__ == '__main__':
    gerar_graficos_brutos()
