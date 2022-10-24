/****** Script for SelectTopNRows command from SSMS  ******/
SELECT distinct COD_CAUS_ANA
      ,[COD_CAUS_ANA]
      ,[DESCR]
      ,[DATA_INIZIO_VALIDITA]
      ,[DATA_FINE_VALIDITA]
      ,[FLAG_CONTO_CASSA]
  FROM [mm_dwa].[dbo].[DEC_CAUS_ANA]
 where DESCR LIKE '%finanziamenti%'