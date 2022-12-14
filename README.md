# Anomaly prioritization task
## 1. Controllo delle librerie
```
init / version_check.py
```
## 2. Sviluppo del dataset partendo dalle informazioni caricate dai file csv
```
input / build_dataset.py
```
Metodo per eseguire e salvare le features
```
run()
```
### 2.1 Parametri della classe **_`input / load_csv.py`_**
<!-- TABLE_GENERATE_START -->

| Parametro             | Default                                         | Commento                                                            |
|-----------------------|-------------------------------------------------|---------------------------------------------------------------------|
| evaluation_csv        | target_not_processed.csv                        |                                                                     |
| operations_csv        | all_operations_db.csv                           |                                                                     |
| subject_csv           | all_subjects.csv                                |                                                                     |
| operations_day_csv    | all_operations_day_db.csv                       | Informazioni addizionali per il discovery day                       |
| accounts_csv          | all_accounts_db.csv                             |                                                                     |
| list_values_csv       | list_values.csv                                 |                                                                     |
| causal_analytical_csv | causale_analitica.csv                           |                                                                     |
| start_date_evaluation | 2021-02-01 00:00:00.001                         | Data di inizio delle evaluation                                     |
| max_months_considered | 19                                              | Numero di mesi: 1 anno + 6 mesi precedenti                          |
| software_list_to_drop | Considerazione solo del Discovery Comportamenti | lista dei software da non considerare (e.g. Discovery Day)          |
| state_list_to_drop    | None                                            | lista degli stati delle valuation da eliminare (e.g. NOT_EVALUATED) |

<!-- TABLE_GENERATE_END -->

### 2.2 Parametri della classe **_`input / build_features.py`_**
<!-- TABLE_GENERATE_START -->

| Parametro                  | Default                       | Commento                                                                                                        |
|----------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------|
| csv_data                   | classe `input / load_csv.py`  |                                                                                                                 |
| max_elements               | None                          | Usato per la fase di testing.                                                                                   |
| prefix                     |                               | Prefissi utilizzati per definire i nomi delle features legate alle operazioni (e.g. tot (i.e. totale)).         |
| months                     | [3, 6]                        |                                                                                                                 |
| range_repetitiveness       | 0.05                          | Range limite(tra 0-1) per misurare la ripetitivit?? dei movimenti.                                               |
| variance_threshold_1       | 0.80                          | Prima soglia utilizzata per eliminare i valori elevanti nel calcolo del numero di operazioni ripetute nei mesi. |
| variance_threshold_2       | 0.75                          | Seconda soglia utilizzata per eliminare i valori nel calcolo del numero di operazioni ripetute nei mesi.        |
| variance_threshold_filiale | 35                            | Soglia per eliminare il numero di versamenti nelle filiari  che si discostano troppo dalla media.               |
| min_age                    | 21                            | Valore minimo per clusterizzare l'et??                                                                           |
| max_age                    | 110                           | Valore massimo per clusterizzare l'et??                                                                          |
| step_age                   | 10                            |                                                                                                                 |
| path_x                     | /data/dataset_x.csv           |                                                                                                                 |
| path_y                     | /data/dataset_y.csv           |                                                                                                                 |
| path_x_evaluated           | /data/dataset_x_evaluated.csv | Percorso per salvare le features delle anomalie con target = 1                                                  |
| causal version             | raggruppamento a 14 (v14)     | Raggruppamento della causali                                                                                    |
| production                 | False                         | True quando dovr?? essere utilizzato in produzione                                                               |

<!-- TABLE_GENERATE_END -->

## 3. Categorizzazione, normalizzazione e gestione degli elementi nulli
```
pre_processing / categorization.py
```
Metodo per eseguire e salvare le features processate
```
Categorization(x_dataset).run()
```
Altrimenti utilizzando il valore di default salvato in /data/dataset
```
Categorization().run()
```
Il risultato prodotto viene salvato in /data/dataset_split in cui il dataset viene division in train, validation e test

## 4. Training del modello definito in **_`train/models / models_definition.py`_**
```
train / train.py
```
Metodo per addestrare il modello che verr?? salvato in /models (e.g. voting_model.sav)
```
run()
```
## 5. Analisi della distribuzione degli errori
```
test / error_perfomance_estimation.py
```