SELECT COUNT(*) FROM public.queimadas -- TOTAL DE LINHAS DE QUEIMADAS c  -- inpe 
SELECT COUNT(*) FROM public.reservas -- TOTAL DE LINHAS DE QUEIMADAS 50   -- funai 
----------------------
-- atualizacao dos SRIDs 

UPDATE queimadas SET geom = st_setsrid(geom, 4326) WHERE st_srid(geom) = 0;
UPDATE reservas  SET geom = st_setsrid(geom, 4326) WHERE st_srid(geom) = 0;
--------------------------------------------------------
-- TRUNCATE TABLE public.queimadas_brasil_reservas
-- DROP TABLE  public.queimadas_brasil_reservas


-- script para criacao\carga da tabela que contem registros de todas as queimadas que ocorreram dentro
-- de uma reserva indigena
-- foi criado um campo flg ( valor 1 ) indicando 
select
tq.gid as gid_q, tq.datahora, tq.satelite, tq.pais, 
tq.estado, tq.municipio, tq.bioma, 
tq.diasemchuv, tq.precipitac, tq.riscofogo, tq.latitude, tq.longitude, tq.frp, tq.geom as geom_q,
tr.gid as gid_r, 
tr.__gid, tr.terrai_cod, tr.terrai_nom, tr.etnia_nome, 
tr.municipio_, tr.uf_sigla, tr.superficie, tr.fase_ti, tr.modalidade, tr.reestudo_t, tr.cr, tr.faixa_fron, 
tr.undadm_cod, tr.undadm_nom, tr.undadm_sig, tr.dominio_un, tr.geom as geom_r,
1 as flg_q_r
INTO public.queimadas_brasil_reservas
from 
     public.queimadas tq,  -- tabela com um total de 219.455 linhas -- objeto POINT
	 public.reservas  tr   -- tabela com um total de 50      linhas -- objeto MULTIPOLYGON
WHERE
   st_contains(tr.geom, tq.geom) 

---------------------------------------------------------------------------------------------
select count(*) from public.queimadas_brasil_reservas;  -- 34 linhas 

------------------------------------------------------------------------------------------
-- script para inclusao de registros de queimadas que ocorreram FORA das areas de reservas indigenas
-- o campo flg nesse caso fica com valor 0

INSERT INTO public.queimadas_brasil_reservas
select
tq.gid as gid_q, tq.datahora, tq.satelite, tq.pais, 
tq.estado, tq.municipio, tq.bioma, 
tq.diasemchuv, tq.precipitac, tq.riscofogo, tq.latitude, tq.longitude, tq.frp, tq.geom as geom_q,
tr.gid as gid_r, 
tr.__gid, tr.terrai_cod, tr.terrai_nom, tr.etnia_nome, 
tr.municipio_, tr.uf_sigla, tr.superficie, tr.fase_ti, tr.modalidade, tr.reestudo_t, tr.cr, tr.faixa_fron, 
tr.undadm_cod, tr.undadm_nom, tr.undadm_sig, tr.dominio_un, tr.geom as geom_r,
0 as flg_q_r
from 
             public.queimadas tq                                     -- tabela com um total de 219.455 linhas -- objeto POINT
LEFT JOIN 	 public.reservas  tr ON st_contains(tr.geom, tq.geom)    -- tabela com um total de 50 linhas - objeto MULTIPOLYGON ON 
WHERE
  tr.gid is null  -- 219.421
   
------------------------------------------------------------------------------------------------------

