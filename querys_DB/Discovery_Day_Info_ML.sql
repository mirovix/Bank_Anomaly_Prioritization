/****** Script for SelectTopNRows command from SSMS  ******/
SELECT	[NDG],
		[CODE_OPERATION],
		[DATA],
		[CAUSAL],
		[SIGN],
		[COUNTRY],
		[AMOUNT],
		[AMOUNT_CASH],
		[COUNTERPART_SUBJECT_COUNTRY],
		[RESIDENCE_COUNTRY_T],	
		/*[RISK_PROFILE_T],*/
		[RESIDENCE_COUNTRY_E],
		[RISK_PROFILE_E]

FROM (SELECT *
FROM (SELECT DISTINCT [mm_dwa].[dbo].[OPERATION_SUBJECT].[NDG],
				[mm_dwa].[dbo].[OPERATION].[CODE_OPERATION],
				[mm_dwa].[dbo].[OPERATION].[DATE_OPERATION] as 'DATA',
				[mm_dwa].[dbo].[OPERATION].[CAUSAL],
				[mm_dwa].[dbo].[OPERATION].[SIGN],
				[mm_dwa].[dbo].[OPERATION].[COUNTRY],
				[mm_dwa].[dbo].[OPERATION].[AMOUNT],
				[mm_dwa].[dbo].[OPERATION].[AMOUNT_CASH],
				[mm_dwa].[dbo].[OPERATION].[COUNTERPART_SUBJECT_COUNTRY],
				[mm_dwa].[dbo].[OPERATION_SUBJECT].[SUBJECT_TYPE],
				[mm_dwa].[dbo].[SUBJECT].[RESIDENCE_COUNTRY] as 'RESIDENCE_COUNTRY_T',
				[mm_dwa].[dbo].[SUBJECT].[RISK_PROFILE] as 'RISK_PROFILE_T',
				E.[NDG] as 'NDG_E',
				E.[SUBJECT_TYPE] as 'SUBJECT_TYPE_E',
				S.[RESIDENCE_COUNTRY] as 'RESIDENCE_COUNTRY_E',
				S.[RISK_PROFILE] as 'RISK_PROFILE_E'

FROM [mm_dwa].[dbo].[OPERATION] 
	 left join [mm_dwa].[dbo].[OPERATION_SUBJECT] on [mm_dwa].[dbo].[OPERATION].[CODE_OPERATION] = [mm_dwa].[dbo].[OPERATION_SUBJECT].[CODE_OPERATION]
	 left join [mm_dwa].[dbo].[SUBJECT] on [mm_dwa].[dbo].[OPERATION_SUBJECT].[NDG] = [mm_dwa].[dbo].[SUBJECT].[NDG]
	 right join [mm_dwa].[dbo].[OPERATION_SUBJECT] E on [mm_dwa].[dbo].[OPERATION].[CODE_OPERATION] = E.[CODE_OPERATION]
	 left join [mm_dwa].[dbo].[SUBJECT] S on E.[NDG] = S.[NDG]

WHERE /*[mm_dwa].[dbo].[OPERATION].[CODE_OPERATION] = '34210472100110005322' and*/
	  [mm_dwa].[dbo].[OPERATION].[CODE_INTERMEDIARY] = '060459' and
	  [mm_dwa].[dbo].[OPERATION_SUBJECT].[SUBJECT_TYPE] = 'T' and 
	  E.[SUBJECT_TYPE] != 'F'
) t

WHERE (NDG = NDG_E and SUBJECT_TYPE_E = 'T') or (NDG != NDG_E and SUBJECT_TYPE_E = 'E') or (NDG = NDG_E and SUBJECT_TYPE_E = 'E')
) t2

WHERE ((SELECT count(*) FROM [mm_dwa].[dbo].[OPERATION_SUBJECT] E2 where E2.[CODE_OPERATION] = t2.[CODE_OPERATION] and E2.[SUBJECT_TYPE]='E') > 0 and t2.[SUBJECT_TYPE_E] != 'T') or
	  ((SELECT count(*) FROM [mm_dwa].[dbo].[OPERATION_SUBJECT] E2 where E2.[CODE_OPERATION] = t2.[CODE_OPERATION] and E2.[SUBJECT_TYPE]='E') = 0)

ORDER BY [DATA]