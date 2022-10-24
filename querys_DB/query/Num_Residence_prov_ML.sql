/****** Script for SelectTopNRows command from SSMS  ******/
SELECT [RESIDENCE_PROVINCE], count(*) as n

 FROM [mm_dwa].[dbo].[SUBJECT]
 group by [RESIDENCE_PROVINCE]
 order  by n desc