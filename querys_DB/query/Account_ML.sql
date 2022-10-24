/****** Script for SelectTopNRows command from SSMS  ******/
SELECT DISTINCT [mm_dwa].[dbo].[ACCOUNT].[CODE_ACCOUNT]
      ,[mm_dwa].[dbo].[ACCOUNT].[START_DATE]
      ,[mm_dwa].[dbo].[ACCOUNT].[EXPIRE_DATE]
	  ,[mm_dwa].[dbo].[ACCOUNT_SUBJECT].[NDG]
  FROM [mm_dwa].[dbo].[ACCOUNT]
  INNER JOIN  [mm_dwa].[dbo].[ACCOUNT_SUBJECT] ON [mm_dwa].[dbo].[ACCOUNT].[CODE_ACCOUNT] =  [mm_dwa].[dbo].[ACCOUNT_SUBJECT].[CODE_ACCOUNT]
  WHERE [mm_dwa].[dbo].[ACCOUNT_SUBJECT].[CODE_INTERMEDIARY] = '060459'