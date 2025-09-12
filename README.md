# REDEM-Index

Repositório para cálculo e análise de índices de qualidade parlamentar e representação descritiva no Brasil.

## 📋 Visão Geral

O REDEM-Index é um projeto de pesquisa que desenvolve e aplica metodologias para avaliar a qualidade parlamentar e a representação descritiva no sistema político brasileiro. O projeto combina análise de dados parlamentares, métodos estatísticos avançados e ferramentas computacionais para gerar insights sobre o funcionamento do Congresso Nacional.

## 🏗️ Estrutura do Projeto

### 📁 Diretórios Principais

```
redem-index/
├── 📊 data/                    # Dados brutos e processados
├── 📚 docs/                    # Documentação acadêmica (LaTeX)
│   ├── IPP/                    # Documentação técnica do IPP
│   ├── IRD/                    # Documentação técnica do IRD
│   ├── tables/                 # Tabelas de resultados
│   └── main.tex                # Documento principal
├── 🖼️ imagens/                 # Gráficos e visualizações
├── 📈 outputs/                 # Resultados dos cálculos
├── 🐍 src/                     # Código fonte Python
├── 📖 studies/                 # Estudos e análises específicas
└── 🛠️ utils/                   # Utilitários e ferramentas
```

### 🔧 Componentes do Sistema

#### 1. **Extratores de Dados** (`src/extractors/`)
- **`extractor.py`**: Classe principal para extração de dados da API do Congresso
- **`talker.py`**: Interface de comunicação com APIs externas
- Extrai dados de parlamentares, legislaturas, lideranças, comissões e pareceres

#### 2. **Calculadores de Índices** (`src/calculators/`)
- **`base_index.py`**: Classe base para todos os índices
- **`ipp_calculator.py`**: Calculador do Índice de Profissionalização do Parlamentar (IPP)
- **`ird_calculator.py`**: Calculador do Índice de Representação Descritiva (IRD)s

#### 3. **Utilitários** (`utils/`)
- **`api.py`**: Configurações de API
- **`plots.py`**: Funções para visualizações
- **`pca_utils.py`**: Utilitários para Análise de Componentes Principais
- **`utils.py`**: Funções auxiliares gerais

## 🚀 Como Usar

### 0. **Download dos dados**
Os dados podem ser baixados por este [link](https://1drv.ms/u/c/c66f0a8fee5f3530/EbY6beqUzFBOvhIP13exGJgBZ3D3HlD7y13NTU0HjWNWlQ?e=mc3XW7).
Salvar a pasta 'data' dentro da pasta raiz do repositório.

### 1. **Instalação de Dependências**

```bash
python -m venv .venv
pip install -r requirements.txt
```

### 2. **Compilação da Documentação (Opcional)**

Para gerar a documentação em PDF:

```bash
cd docs
pdflatex main.tex
```

### 3. **Extração de Dados (opcional)**

```python
from extraction import main

# Extrair todos os dados do Congresso
main()
```

### 4. **Cálculo do IPP**

```python
from src.calculators.ipp_calculator import IPPCalculator

# Inicializar calculador
ipp_calc = IPPCalculator()

# Calcular índice
df_resultado = ipp_calc.calculate(df_dados)
```

### 5. **Cálculo do IRD**

```python
from src.calculators.equality_index_calculator import EqualityIndexCalculator

# Inicializar calculador
equality_calc = EqualityIndexCalculator()

# Definir proporções populacionais esperadas
pop_proportions = {'F': 0.5, 'M': 0.5}

# Calcular pesos e índice
weights, adjustment_index = equality_calc.compute_individual_weights(
    df_categorias, pop_proportions
)
```

## 📚 Estudos e Documentação

### **IPP (Índice de Profissionalização do Parlamentar)**
- `studies/ipp/ipp_v2.ipynb`: Implementação principal do IPP
- `studies/ipp/analise_fatorial_v2.ipynb`: Análise fatorial exploratória
- `studies/ipp/analyse.ipynb`: Análises complementares
- `docs/IPP/ipp.tex`: Documentação técnica completa em LaTeX

### **Representação Descritiva (IRD)**
- `studies/ird/indice_representacao_descirtiva.ipynb`: Cálculo do IRD
- `studies/ird/REDEM___Dimensao_Representação.pdf`: Documentação teórica
- `docs/IRD/ird.tex`: Documentação técnica detalhada em LaTeX

### **Estudos Específicos**
- **Evangélicos**: Análise da representação parlamentar evangélica
- **Fake News**: Estudo sobre desinformação e parlamentares
- **Coesão Interna**: Índice de coesão interna dos partidos

## 📊 Dados Disponíveis

### **Fontes Principais:**
- **TSE**: Dados eleitorais e candidatos
- **API do Congresso**: Informações parlamentares, legislaturas, comissões
- **IBGE**: Dados demográficos da população brasileira

### **Tipos de Dados:**
- Perfis parlamentares detalhados
- Histórico de mandatos e cargos
- Participação em comissões e lideranças
- Pareceres de relatoria
- Dados demográficos (gênero, raça)


## 📝 Documentação

### **Documentação Acadêmica (LaTeX):**
- `docs/main.tex`: Documento principal que integra todas as seções
- `docs/IRD/ird.tex`: Documentação técnica detalhada do Índice de Representação Descritiva
- `docs/IPP/ipp.tex`: Documentação técnica completa do Índice de Qualidade Parlamentar
- `docs/tables/`: Tabelas de resultados para publicação acadêmica

### **Notebooks de Análise:**
- `dashboard.ipynb`: Dashboard interativo para visualização dos índices
- `indices_calculation.ipynb`: Cálculos principais e implementação dos índices
- `cfa.ipynb`: Análise fatorial confirmatória

## 🤝 Contribuição

Para contribuir com o projeto:

1. **Fork** o repositório
2. **Crie** uma branch para sua feature
3. **Implemente** suas mudanças
4. **Teste** com os dados disponíveis
5. **Submeta** um pull request

## 📄 Licença

Este projeto é desenvolvido para fins de pesquisa acadêmica. Consulte os autores para uso comercial ou redistribuição.

<!-- ## 👥 Autores

- **Acácio Telechi**: acaciotelechi@gmail.com
- **Nilton -->

## 📞 Contato

Para dúvidas ou colaborações, entre em contato com os autores ou abra uma issue no repositório.

---

*REDEM-Index: Ferramentas para análise da qualidade parlamentar e representação descritiva no Brasil*
