import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from pathlib import Path

# ==============================================================================
# CONFIGURAÇÕES E PARÂMETROS
# ==============================================================================
ELEMENTS_WAVELENGTHS = {
    'N': [742.36, 744.23, 746.83, 821.63, 868.02],
    'P': [213.62, 214.91, 253.56, 255.33],
    'K': [766.49, 769.90, 404.72]
}

ELEMENTS_COLORS = {'N': '#3498db', 'P': '#e67e22', 'K': '#9b59b6'}

SAVGOL_WINDOW = 11      
SAVGOL_POLY = 2         
BASELINE_DEGREE = 2    
BASELINE_ITER = 10     
PEAK_PROMINENCE_FRAC = 0.02 
PEAK_TOLERANCE = 0.6   

# ==============================================================================
# FUNÇÃO DE LEITURA ADAPTATIVA
# ==============================================================================
def carregar_arquivo_esf(caminho_arquivo):
    encodings = ['utf-8', 'utf-16', 'latin-1']
    for enc in encodings:
        try:
            with open(caminho_arquivo, 'r', encoding=enc, errors='ignore') as f:
                linhas = f.readlines()
            
            if not linhas: continue
                
            dados_num = []
            for linha in linhas:
                linha_txt = linha.strip()
                if not linha_txt: continue
                
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
            
    raise ValueError("Não foi possível extrair duas colunas numéricas válidas do arquivo.")

# ==============================================================================
# FUNÇÕES DE PROCESSAMENTO MATEMÁTICO
# ==============================================================================
def filtrar_ruido(intensidade, janela=SAVGOL_WINDOW, ordem=SAVGOL_POLY):
    if janela % 2 == 0: janela += 1  
    if len(intensidade) <= janela: return intensidade
    return savgol_filter(intensidade, window_length=janela, polyorder=ordem)

def remover_baseline(intensidade, grau=BASELINE_DEGREE, iteracoes=BASELINE_ITER):
    base = intensity = intensidade.copy()
    n = len(intensidade)
    if n == 0: return intensidade, np.zeros_like(intensidade)
        
    x_escalonado = np.linspace(-1, 1, n)
    for _ in range(iteracoes):
        p = np.polyfit(x_escalonado, base, grau)
        fit = np.polyval(p, x_escalonado)
        base = np.minimum(base, fit)
        
    baseline_estimado = np.polyval(np.polyfit(x_escalonado, base, grau), x_escalonado)
    espectro_corrigido = intensidade - baseline_estimado
    return np.clip(espectro_corrigido, 0, None), baseline_estimado

def normalizar_por_area(comprimento_onda, intensidade):
    area_total = np.trapezoid(intensidade, comprimento_onda)
    return intensidade / area_total if area_total != 0 else intensidade

def identificar_picos_npk(comprimentos_onda, intensidades, tolerancia=PEAK_TOLERANCE, prom_frac=PEAK_PROMINENCE_FRAC):
    prominencia_min = np.max(intensidades) * prom_frac
    picos_idx, _ = find_peaks(intensidades, prominence=prominencia_min, distance=4)
    picos_identificados = []
    
    for idx in picos_idx:
        w_pico, i_pico = comprimentos_onda[idx], intensidades[idx]
        melhor_elemento, linha_ref, menor_diff = None, None, float('inf')
        
        for elemento, linhas in ELEMENTS_WAVELENGTHS.items():
            for linha in linhas:
                diff = abs(w_pico - linha)
                if diff <= tolerancia and diff < menor_diff:
                    menor_diff, melhor_elemento, linha_ref = diff, elemento, linha
                    
        if melhor_elemento:
                    picos_identificados.append({
                        'index': idx, 'elemento': melhor_elemento, 'w_medido': w_pico,
                        'w_referencia': linha_ref, 'intensidade': i_pico, 'desvio': menor_diff
                    })
            
    return picos_identificados, picos_idx

# ==============================================================================
# PIPELINE PRINCIPAL (ESPELHAMENTO EXATO DE DIRETÓRIOS)
# ==============================================================================
def processar_espectros_libs_em_lote():
    print("=" * 70)
    print("INICIANDO PROCESSAMENTO AUTOMATIZADO COM ESPELHAMENTO DE DIRETÓRIOS")
    print("=" * 70)
    
    # '.' aponta para a pasta EMBRAPA onde o script está sendo executado
    pasta_raiz = Path(".")
    pasta_resultados = pasta_raiz / "RESULTADOS_LIBS"
    
    # Localiza todos os arquivos .esf nas subpastas, ignorando a pasta de saída
    arquivos_esf = [f for f in pasta_raiz.rglob("*.esf") if "RESULTADOS_LIBS" not in f.parts and "backup" not in f.parts]
    
    if not arquivos_esf:
        print("[AVISO] Nenhum arquivo .esf encontrado nas subpastas.")
        return

    print(f"Encontrado(s) {len(arquivos_esf)} arquivo(s) .esf para processar.\n")
    
    for caminho_arquivo in arquivos_esf:
        nome_arquivo = caminho_arquivo.name
        nome_base = caminho_arquivo.stem
        
        # Pega a estrutura exata de subpastas (ex: NPK 10-10-10/UV/Maxgreen/500ns)
        caminho_relativo = caminho_arquivo.relative_to(pasta_raiz).parent
        
        # Cria a pasta correspondente dentro de RESULTADOS_LIBS
        dir_destino = pasta_resultados / caminho_relativo
        dir_destino.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Carga e Validação
            wl, intens_bruta = carregar_arquivo_esf(str(caminho_arquivo))
            print(f"-> Espelhando: RESULTADOS_LIBS / {caminho_relativo} / {nome_arquivo}")
            
            # 2. Processamento Matemático
            intens_suave = filtrar_ruido(intens_bruta)                         
            intens_sem_baseline, baseline_est = remover_baseline(intens_suave) 
            intens_normalizada = normalizar_por_area(wl, intens_sem_baseline)  
            
            picos_npk, _ = identificar_picos_npk(wl, intens_normalizada)
            
            # 3. Salvando Tabelas CSV diretamente na pasta espelhada
            df_saida = pd.DataFrame({
                'comprimento_onda_nm': wl, 
                'intensidade_bruta': intens_bruta,
                'intensidade_suavizada': intens_suave,
                'baseline_estimado': baseline_est, 
                'intensidade_pre_processada': intens_normalizada
            })
            df_saida.to_csv(dir_destino / f"{nome_base}_processado.csv", index=False)
            
            if picos_npk:
                pd.DataFrame(picos_npk).to_csv(dir_destino / f"{nome_base}_picos_identificados.csv", index=False )
            
            # 4. Construção do Gráfico diretamente na pasta espelhada
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            ax1.plot(wl, intens_suave, color='#7f8c8d', alpha=0.7, label='Espectro Suavizado')
            ax1.plot(wl, baseline_est, color='#e74c3c', linestyle='--', linewidth=1.5, label='Baseline Estimada')
            ax1.set_title(f"Remoção de Baseline: {nome_arquivo}", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Intensidade Absoluta")
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            ax2.plot(wl, intens_normalizada, color='#2c3e50', linewidth=1, label='Final: Área Normalizada')
            
            elementos_vistos = set()
            for pico in picos_npk:
                elem = pico['elemento']
                label = f"Linha de {elem}" if elem not in elementos_vistos else ""
                elementos_vistos.add(elem)
                
                ax2.axvline(x=pico['w_medido'], color=ELEMENTS_COLORS[elem], linestyle=':', alpha=0.6)
                ax2.scatter(pico['w_medido'], pico['intensidade'], color=ELEMENTS_COLORS[elem], s=45, zorder=5, label=label)
                
                texto = f"{elem}\n{pico['w_medido']:.1f}"
                ax2.annotate(texto, xy=(pico['w_medido'], pico['intensidade']), xytext=(4, 4),
                             textcoords='offset points', fontsize=8, fontweight='bold', color=ELEMENTS_COLORS[elem])
            
            ax2.set_title("Espectro Corrigido com Mapeamento Químico NPK", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Comprimento de Onda (nm)")
            ax2.set_ylabel("Intensidade Relativa")
            ax2.legend(loc='upper right')
            ax2.grid(True, linestyle=':', alpha=0.6)
            
            plt.tight_layout()
            plt.savefig(dir_destino / f"{nome_base}_espectro.png", dpi=200)
            plt.close()
            
        except Exception as e:
            print(f"   [ERRO] Falha ao processar {nome_arquivo}. Motivo: {str(e)}")
            
    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO! Nova estrutura criada em 'RESULTADOS_LIBS'")
    print("=" * 70)

if __name__ == '__main__':
    processar_espectros_libs_em_lote()
