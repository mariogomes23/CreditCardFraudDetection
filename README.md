# Detecção de Fraudes em Transações de Cartão de Crédito

Este projeto utiliza técnicas de aprendizado de máquina para detectar fraudes em transações financeiras de cartão de crédito. O foco principal é a **detecção de anomalias**, utilizando modelos como o **Isolation Forest** e o **One-Class SVM**. O objetivo é identificar transações suspeitas (fraudulentas) a partir de um conjunto de dados contendo transações normais e fraudulentas.

## Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Requisitos](#requisitos)
- [Como Executar](#como-executar)
- [Saídas](#saídas)
- [Relatórios de Desempenho](#relatórios-de-desempenho)
- [Contribuições](#contribuições)
- [Licença](#licença)

## Visão Geral

O dataset utilizado é composto por transações de cartão de crédito, com a variável `Class` indicando se uma transação é normal (0) ou fraudulenta (1). O código aplica dois modelos para detecção de fraudes:

1. **Isolation Forest**: Um modelo baseado em árvores de decisão para detecção de anomalias.
2. **One-Class SVM**: Um modelo baseado em máquinas de vetores de suporte para identificar padrões atípicos.

Além da detecção de fraudes, o código também gera visualizações que ajudam a entender a distribuição das transações e os valores das transações.

## Estrutura do Projeto

```bash
- src/
  - app.py              # Código principal do projeto
  - data/
    - creditcard.csv    # Dataset de transações de cartão de crédito (necessário para execução)
  - output/
    - class_distribution.png  # Gráfico de distribuição das classes (normais vs fraudulentas)
    - transaction_amount_distribution.png  # Gráfico de distribuição do valor das transações
