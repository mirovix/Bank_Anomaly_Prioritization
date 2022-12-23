# Informazioni utili per l'estrattore:
1. definire il file di configurazione (config.py) per la connessione al db e per definire i vari percorsi di salvataggio.
2. definire le query (queries.sql) separate dal delimitatore ';'.
3. i file (csv) sono salvati nella cartella output.

# Librerie da installare, l'ultima versione dovrebbe andare bene":
1. pandas
2. sqlalchemy
3. PyMySql
4. cx_Oracle

# Avviare l'estrattore:
1. aprire un cmd nella cartella csv_extractor_dir
2. eseguire Python csv_extractor.py
3. in alternativa se si vuole utilizzare un proprio venv: C:\^path_to_venv^\Scripts\python.exe csv_extractor.py
   (e.g. C:\workspace\AnomalyPrioritization\venv_3.9\Scripts\python.exe csv_extractor.py)