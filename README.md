# SensorLab AI

SensorLab AI è un'applicazione didattica per collegare sensori Arduino a modelli di machine learning e, in modo controllato, usare le previsioni per pilotare attuatori.

Il progetto funziona subito senza hardware grazie a un simulatore. Quando Arduino sarà disponibile, la sorgente può essere cambiata in `seriale` dalla pagina **Configurazione**, senza modificare il codice dell'applicazione.

## Funzioni principali

- monitoraggio in tempo reale di temperatura, umidità, vibrazione e altri canali configurabili;
- sorgente dati simulata oppure seriale Arduino;
- caricamento di file CSV e generazione di un dataset dimostrativo;
- training con **Linear Regression** per problemi di regressione;
- training con **XGBoost** per regressione o classificazione;
- salvataggio, attivazione e uso dei modelli addestrati;
- regole `previsione → comando attuatore` con modalità simulata attiva per impostazione predefinita;
- API documentate automaticamente da FastAPI.

> Sicurezza: l'invio fisico di comandi agli attuatori è disabilitato per impostazione predefinita. Prima di abilitarlo, verificare cablaggio, alimentazione, limiti elettrici e presenza di un arresto manuale.

## Avvio rapido

Richiede Python 3.11 o successivo.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Aprire `http://127.0.0.1:8000`. La documentazione API è disponibile in `http://127.0.0.1:8000/docs`.

Per eseguire i test:

```powershell
python -m pip install -r requirements-dev.txt
pytest
```

## Percorso didattico consigliato

1. Lasciare la sorgente su **Simulazione** e osservare i segnali nel monitor.
2. Dal **Laboratorio modelli**, generare il dataset demo oppure caricare un CSV.
3. Scegliere la colonna target e le feature, quindi addestrare un modello.
4. Leggere le metriche: MAE, RMSE e R² per regressione; accuracy, precision, recall e F1 per classificazione.
5. Attivare il modello e verificare le previsioni in tempo reale.
6. Creare una regola attuatore e controllare gli eventi in modalità simulata.
7. Solo dopo questi passaggi, configurare Arduino e testare la connessione seriale.

## Formato del dataset

Il file deve essere CSV con intestazioni nella prima riga. Ogni riga rappresenta un'osservazione; le feature devono essere numeriche. La colonna target può essere numerica (regressione) oppure contenere classi (classificazione con XGBoost).

Esempio:

```csv
temperature,humidity,vibration,target
22.4,48.2,0.12,0
27.1,60.5,0.35,1
```

Un file più ampio è disponibile in [`examples/sensor_data_example.csv`](examples/sensor_data_example.csv).

### Compatibilità dei modelli

| Modello | Regressione | Classificazione |
| --- | --- | --- |
| Linear Regression | sì | no |
| XGBoost | sì | sì |

Linear Regression non produce classi. Per evitare risultati didatticamente ambigui, l'API rifiuta questa combinazione e indica di scegliere XGBoost.

## Protocollo Arduino predisposto

La connessione seriale è intenzionalmente esplicita: porta, baud rate, intervallo di lettura e mapping dei campi sono modificabili dalla UI. Un esempio completo è in [`examples/arduino.example.json`](examples/arduino.example.json).

Arduino invia una riga JSON terminata da `\n` per ogni campione:

```json
{"temperature": 23.8, "humidity": 51.2, "vibration": 0.18, "light": 420}
```

Nell'app le stesse grandezze possono avere nomi più descrittivi, ad esempio `temperature_c`. Il campo `source_key` della configurazione associa esplicitamente `temperature_c` alla chiave Arduino `temperature`; l'esempio JSON contiene già questo mapping.

Quando l'esecuzione reale è abilitata, l'app può inviare una riga JSON dello stesso tipo:

```json
{"actuator": "fan", "value": 1}
```

Uno sketch completo, senza librerie esterne, è disponibile in
[`arduino/meraviglia_sensor_node.ino`](arduino/meraviglia_sensor_node.ino). Genera valori simulati finché le quattro funzioni di lettura non vengono sostituite e accetta esclusivamente il comando per il LED integrato, così è possibile provare l'intero flusso senza collegare carichi esterni.

Sketch minimo di riferimento:

```cpp
void setup() {
  Serial.begin(115200);
}

void loop() {
  float temperature = 23.5; // sostituire con il sensore reale
  float humidity = 50.0;
  float vibration = 0.10;
  float light = 420.0;

  Serial.print("{\"temperature\":");
  Serial.print(temperature);
  Serial.print(",\"humidity\":");
  Serial.print(humidity);
  Serial.print(",\"vibration\":");
  Serial.print(vibration);
  Serial.print(",\"light\":");
  Serial.print(light);
  Serial.println("}");
  delay(1000);
}
```

Il parser lato server ignora righe non valide e restituisce un errore leggibile nel test di connessione. I nomi inviati devono corrispondere alle feature usate dal modello attivo.

## Regole attuatori

Una regola confronta la previsione del modello con una soglia, per esempio:

```text
prediction > 30  →  fan = 1
```

Ogni regola può essere abilitata o disabilitata. Finché la modalità reale non è attiva, l'evento viene registrato ma nessun byte viene inviato alla porta seriale. È consigliato aggiungere isteresi e limiti temporali nel firmware per attuatori critici.

I comandi sono eseguiti sui cambi di stato: `command_on` quando la condizione diventa vera e `command_off` quando torna falsa, quando la regola viene disabilitata o prima di fermare l'acquisizione. In modalità reale una regola senza `command_off` non viene attivata, così l'app non può lasciare intenzionalmente un attuatore acceso senza una via di spegnimento.

## Struttura del progetto

```text
app/                  backend FastAPI e interfaccia statica
arduino/              sketch didattico pronto per Arduino IDE
examples/             configurazioni e dataset dimostrativi
data/                 configurazione, dataset, modelli e log runtime
tests/                test automatici
requirements.txt      dipendenze Python
```

## Note per passare all'hardware reale

- installare il driver USB corretto e individuare la porta (`COM3`, `/dev/ttyACM0`, ecc.);
- chiudere il Serial Monitor dell'IDE Arduino: una porta seriale normalmente può essere aperta da un solo processo;
- usare lo stesso baud rate nel firmware e nell'app;
- assicurarsi che ogni campione sia una singola riga JSON valida;
- testare prima con attuatori scollegati o in modalità simulata;
- non collegare motori, pompe o relè direttamente ai pin: usare driver e protezioni adeguati.

## Limiti intenzionali

È una base didattica, non un sistema industriale o safety-critical. Non include autenticazione, ridondanza, watchdog hardware, tracciabilità certificata o controllo real-time deterministico.

