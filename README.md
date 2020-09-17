# ClassifyStructuralTweets
## Introduzione
Progetto svolto per l'esame di Data Mining presso l'università di pisa, lo stesso è volto ad individuare problemi nell'infrastruttura cittadina che portano all'allagamento stradale.

Per individuare tali eventi è stato addestrato un classificatore, sfruttando tecniche di text mining, sui tweets provenienti da specifiche zone e che rispettassero determinate parole chiave.

La classificazione è binaria:
* Strutturali
* Non Strutturali

Maggiori dettagli presenti nella cartella documentazione.

## Strumenti
Il progetto è stato svolto utilizzando:
* **GetOldTweet Library** per l'estrazione dei tweet
* **Weka** per la fase di pre-processing e classificazione
* **NetBeans** IDE con cui eseguire il progetto

## Eseguibili
Sono presenti cinque eseguibili:

* **classifier/DownloadTweets** per l'estrazione dei tweet da poter poi classificare a mano
* **classifier/BuildClassifier** Exp.1 - Cerca di gestire il dataset sbilanciato
* **classifier/BuildClassifierSameTweetsPerClassAndKeywords** Exp.2 - Associa egual numero di tweet strutturali e non per ogni parola
* **classifier/BuildClassifierConstantNumberOfNSTweets** Exp.3 - Usa n tweets per parola dei non strutturali e tutte le parole degli strutturali
* **classifierstructuraltweets/MainApp** Fa partire il tool che permette di scaricare e classificare nuovi tweet data una posizione, un range di date e selezionando il classificatore voluto

