-- consultas PI

-- 1ª CONSULTA CONJUNTO ( ST_uniON )  -- operador conjunto 

SELECT 
st_union(t1.geom_q), t1.terrai_nom
FROM 
	public.queimadas_brasil_reservas t1
WHERE
 t1.terrai_nom IS NOT NULL	
GROUP BY 
t1.terrai_nom
--------------------------------------------


--------------------------------------------
-- 2ª apresenta as areas em km2 das reservas por UF  ( OPERADOR metrico )
SELECT 
   uf_sigla, 
   municipio,   
   st_area(geom_r::geography)/1000 as area
FROM 
     public.queimadas_brasil_reservas
where 
     geom_r is not null
GROUP BY 
   uf_sigla,
   municipio,
   area
;

------------------------------------------------
-- 3ª - consulta topologica 

SELECT 
 distinct 
  st_dimension(geom_r) as dimensao_reserva, 
  geometrytype(geom_r) as objeto_reserva,
  st_dimension(geom_q) as dimensao_queimada,
  geometrytype(geom_q) as objeto_queimada
FROM 
 public.queimadas_brasil_reservas
where 
 geom_r is not null
;

------------------------------------------------
-- 4ª consulta topologico 
SELECT 
   bioma, 
   riscofogo,
   diasemchuv
FROM 
   public.queimadas_brasil_reservas
WHERE 
       st_contains( geom_r ,geom_q) 
   and estado='MATO GROSSO';
   
------------------------------------------------
-- 5ª  operador metrico 
SELECT 
    uf_sigla,
	bioma,
	municipio,
	terrai_nom,
	ST_Perimeter(geom_r::geography)/1000 as perimetro
	FROM 
	 public.queimadas_brasil_reservas
where
       riscofogo > 0 
   and geom_r is not null
group by 
uf_sigla, 
bioma,
municipio,
terrai_nom,
perimetro
order by 1,2,3,4

--------------------------------------------------



