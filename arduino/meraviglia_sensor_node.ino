/*
 * SensorLab AI - nodo Arduino dimostrativo
 *
 * - Invia un campione JSON per riga a 115200 baud.
 * - Accetta il solo comando didattico per il LED integrato:
 *     {"actuator":"led","value":1}
 * - Non richiede librerie esterne.
 *
 * I valori dei sensori sono simulati. Sostituire readTemperature(),
 * readHumidity(), readVibration() e readLightLux() quando l'hardware sarà disponibile.
 */

const unsigned long SAMPLE_INTERVAL_MS = 1000;
const size_t COMMAND_BUFFER_SIZE = 96;

char commandBuffer[COMMAND_BUFFER_SIZE];
size_t commandLength = 0;
unsigned long lastSampleAt = 0;

float readTemperature() {
  // Oscillazione lenta fra circa 21 e 29 °C.
  return 25.0 + 4.0 * sin(millis() / 12000.0);
}

float readHumidity() {
  // Oscillazione lenta fra circa 43 e 57 %.
  return 50.0 + 7.0 * sin(millis() / 17000.0 + 1.2);
}

float readVibration() {
  // Segnale dimostrativo 0.05 - 0.35.
  return 0.20 + 0.15 * sin(millis() / 3500.0);
}

float readLightLux() {
  // Illuminamento dimostrativo fra circa 120 e 880 lux.
  return 500.0 + 380.0 * sin(millis() / 9000.0 + 0.6);
}

void emitSensorSample() {
  Serial.print(F("{\"temperature\":"));
  Serial.print(readTemperature(), 2);
  Serial.print(F(",\"humidity\":"));
  Serial.print(readHumidity(), 2);
  Serial.print(F(",\"vibration\":"));
  Serial.print(readVibration(), 3);
  Serial.print(F(",\"light\":"));
  Serial.print(readLightLux(), 0);
  Serial.println(F("}"));
}

bool commandIsForLed(const char *line) {
  return strstr(line, "\"actuator\":\"led\"") != NULL ||
         strstr(line, "\"actuator\": \"led\"") != NULL;
}

void handleCommand(const char *line) {
  if (!commandIsForLed(line)) {
    Serial.println(F("{\"type\":\"command_ack\",\"ok\":false,\"error\":\"unknown actuator\"}"));
    return;
  }

  const bool turnOn = strstr(line, "\"value\":1") != NULL ||
                      strstr(line, "\"value\": 1") != NULL ||
                      strstr(line, "\"value\":true") != NULL;
  const bool turnOff = strstr(line, "\"value\":0") != NULL ||
                       strstr(line, "\"value\": 0") != NULL ||
                       strstr(line, "\"value\":false") != NULL;

  if (!turnOn && !turnOff) {
    Serial.println(F("{\"type\":\"command_ack\",\"ok\":false,\"error\":\"value must be 0 or 1\"}"));
    return;
  }

  digitalWrite(LED_BUILTIN, turnOn ? HIGH : LOW);
  Serial.print(F("{\"type\":\"command_ack\",\"ok\":true,\"actuator\":\"led\",\"value\":"));
  Serial.print(turnOn ? 1 : 0);
  Serial.println(F("}"));
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = (char) Serial.read();

    if (incoming == '\r') {
      continue;
    }

    if (incoming == '\n') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        handleCommand(commandBuffer);
        commandLength = 0;
      }
      continue;
    }

    if (commandLength < COMMAND_BUFFER_SIZE - 1) {
      commandBuffer[commandLength++] = incoming;
    } else {
      commandLength = 0;
      Serial.println(F("{\"type\":\"command_ack\",\"ok\":false,\"error\":\"command too long\"}"));
    }
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  lastSampleAt = millis();
}

void loop() {
  readSerialCommands();

  const unsigned long now = millis();
  if (now - lastSampleAt >= SAMPLE_INTERVAL_MS) {
    lastSampleAt = now;
    emitSensorSample();
  }
}

