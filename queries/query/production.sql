SELECT *
FROM Subject_ML
WHERE NDG in (SELECT NDG
              FROM Operations_Subjects_ML os join Operations_ML o on os.CODE_OPERATION = o.CODE_OPERATION
              WHERE os.CODE_OPERATION in (SELECT CODE_OPERATION FROM Operations_Subjects_ML WHERE NDG in;
SELECT *
FROM Account_ML
WHERE CODE_ACCOUNT in (SELECT CODE_ACCOUNT FROM Account_ML WHERE NDG in;
SELECT o.*
FROM Operations_Subjects_ML os join Operations_ML o on os.CODE_OPERATION = o.CODE_OPERATION
WHERE NDG in;
SELECT os.*
FROM Operations_Subjects_ML os join Operations_ML o on os.CODE_OPERATION = o.CODE_OPERATION
WHERE os.CODE_OPERATION in (SELECT CODE_OPERATION FROM Operations_Subjects_ML WHERE NDG in