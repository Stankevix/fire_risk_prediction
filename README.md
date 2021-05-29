# RF_Prediction

Com impactos ambientais e sociais, as ocorrências de queimadas vem aumentando nos últimos dois anos conforme focos monitorados desde 1998 pelo Instituto Nacional de Pesquisas Espaciais (INPE). Esses episódios de queima da vegetação nativa para desenvolvimento rural, urbano, na agricultura e pecuária aumentam a produção de poluentes atmosféricos e acabam alterando a biodiversidade brasileira.

A medida risco fogo (RF) mensura a probabilidade de ocorrência de focos de incêndio e é baseada em dados meteorológicos diários obtidos por sensoriamento remoto espacial desenvolvida pelo INPE. 

O Brasil é composto pelos biomas amazônia, cerrado, caatinga, pantanal, mata atlântica e pampa. Quando aplicada a métrica RF  em dados reais espaciais de queimadas e meteorológicos do ano de 2020/2021, segmentados por biomas e considerando variáveis exógenas temporais como mês e hora, pode-se observar comportamentos distintos em cada um deles. 

Juntamente com os resultados dos testes estatísticos e análises de tendências, que apontam correlação com maior quantidade de variáveis que compõem este estudo, e o fato do pantanal ter tido quase um terço da sua área desvatada pelas queimadas em 2020, este bioma foi selecionado para aplicação de métodos de regressão baseados em árvores de decisão para predizer seus valores de RF. O erro médio quadrático resultante foi de 0,14 reafirmando importância de considerar variáveis exógenas temporais permitindo predizer a suscetibilidade de queimadas no pantanal.


<p align="center">
  <img src="https://github.com/Stankevix/risco_fogo_prediction/blob/main/Docs/random_forest.png" alt="Sublime's custom image"/>
</p>


## Team:

* Ana Paula Fernandes Lucio Menezes 
* Gabriel Stankevix Soares
* Heron Carlos Gonçalves
* Isabela  Fernanda Capetti

## Status

Working in progress

## Codes

* postgres_conn: Arquivo com a classe de conexão com postgresql
* feature_eng: Conexão com banco de dados e extração de dados conforme query.
* models: Ensemble de modelos de regressão e arvore de decisão


## Article

Overleaf link

https://www.overleaf.com/project/6048d9c3a97b1e652fac912b

## Database
Banco de dados espacial do Brasil, reservas indígenas e incêndios florestais no Brasil. Os dados foram armazenados em Postgresql e a integração com python para o desenvolvimento do modelo de predição.

## Limitations

Dados apenas de 2020, necessario explorar um historico maior de dados.