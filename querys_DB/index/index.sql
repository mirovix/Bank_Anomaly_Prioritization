CREATE INDEX index_operations_ML ON OPERATION (CODE_OPERATION);
CREATE INDEX index_operations_subjects_ML ON OPERATION_SUBJECT (NDG ,CODE_OPERATION);
CREATE INDEX index_operations_subjects_ML_op ON OPERATION_SUBJECT (CODE_OPERATION);
CREATE INDEX index_subject_ML ON SUBJECT (NDG)