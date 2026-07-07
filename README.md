Processamento de Dados LIBS - Caracterização de Fertilizantes NPK 🔬🌾

Este repositório contém os scripts em Python desenvolvidos durante o meu estágio obrigatório em Engenharia na Embrapa Instrumentação (São Carlos - SP). O conjunto de ferramentas aqui presente foi criado para automatizar o pré-processamento, a análise estatística e a modelagem quimiométrica de dados adquiridos via Espectroscopia de Plasma Induzido por Laser (LIBS).

O objetivo do projeto é viabilizar uma metodologia de triagem rápida e robusta para a detecção de adulterações comerciais em fertilizantes primários (NPK), empregando técnicas de aprendizado de máquina não supervisionado e análise de imagens.
📂 Estrutura do Repositório

Os arquivos descritos abaixo compõem a pipeline completa de análise, desde o tratamento do sinal bruto até a extração de insights morfológicos e espectrais:

    processador_libs.py: Módulo base focado no pré-processamento individual das assinaturas espectrais. Inclui algoritmos para correção de linha de base (baseline correction), supressão de ruído térmico e normalização de área, preparando o sinal para a etapa analítica.

    processador_lote.py: Script de automação projetado para a leitura e processamento em massa (batch processing) de centenas de arquivos de espectros exportados pelo software do espectrômetro, garantindo agilidade no tratamento do banco de dados.

    grafico_dados_brutos.py: Ferramenta de visualização de dados responsável por plotar os espectros LIBS originais, permitindo uma inspeção visual rápida das linhas de emissão atômica (ex: Fósforo, Potássio) antes do tratamento algorítmico.

    analise_pca.py: Módulo central de quimiometria. Aplica a Análise de Componentes Principais (PCA) sobre as matrizes de dados espectrais já corrigidas. Este script é fundamental para reduzir a dimensionalidade dos dados e gerar gráficos de escores (score plots), evidenciando o agrupamento/separação entre formulações comerciais autênticas e possíveis outliers (fraudes).

    tratamento_de_ks: Script direcionado ao estudo morfológico e granulométrico das partículas pré-ablação. Aplica o teste estatístico de Kolmogorov-Smirnov para validar as curvas de distribuição do raio equivalente dos grânulos processados via microscopia óptica.

🛠️ Tecnologias e Bibliotecas Utilizadas

Para executar os scripts deste repositório, recomenda-se a criação de um ambiente virtual Python com as seguintes bibliotecas principais:

    pandas - Para estruturação e manipulação das matrizes de dados.

    numpy - Para operações matemáticas e vetoriais de alto desempenho.

    scikit-learn - Para a aplicação do modelo de Análise de Componentes Principais (PCA).

    matplotlib / seaborn - Para a renderização dos gráficos espectrais e de dispersão (clusters).

    scipy - Para a execução dos testes estatísticos de distribuição (Kolmogorov-Smirnov).

👨‍💻 Autor

    Tomás Marchini Tassi Granja

    Universidade Federal de São Carlos (UFSCar)

    Estagiário - Grupo de Óptica e Fotônica da Embrapa Instrumentação
