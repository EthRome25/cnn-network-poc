# Calimero ML Model API

REST API do zarządzania modelami ML w kontekście Calimero.

## 🚀 Uruchomienie

```bash
cd /home/nor/projekty/calimero-client
pnpm server
```

Serwer uruchomi się na: **http://localhost:3001**

## 📡 Endpointy

### 1. GET `/` - Informacje o API

Zwraca podstawowe informacje o API.

```bash
curl http://localhost:3001/
```

**Odpowiedź:**
```json
{
  "name": "Calimero ML Model API",
  "version": "1.0.0",
  "endpoints": {
    "GET /": "API information",
    "GET /health": "Health check",
    "GET /model": "Get current ML model",
    "POST /model": "Upload new ML model"
  },
  "context": {
    "contextId": "DUYLik9nx5pvhXkshA9dCG4ya52mqHrM5TX3rJxw1LGS",
    "applicationId": "HELDXwknx9tVnj3JKfa3EMyGB9JEsApeijVHzKn5cRVX"
  }
}
```

---

### 2. GET `/health` - Health Check

Sprawdza czy serwer działa.

```bash
curl http://localhost:3001/health
```

**Odpowiedź:**
```json
{
  "success": true,
  "status": "healthy",
  "context": {
    "contextId": "DUYLik9nx5pvhXkshA9dCG4ya52mqHrM5TX3rJxw1LGS",
    "nodeUrl": "http://localhost:2528"
  }
}
```

---

### 3. GET `/model` - Pobierz obecny model ML

Zwraca informacje o obecnym modelu ML przechowywanym w kontekście Calimero.

```bash
curl http://localhost:3001/model
```

**Odpowiedź:**
```json
{
  "success": true,
  "data": {
    "id": "model:API Test Model:3.0.0",
    "name": "API Test Model",
    "version": "3.0.0",
    "description": "Model uploaded via REST API",
    "model_type": "neural_network",
    "file_size": 28,
    "file_data": "QVBJIHRlc3QgbW9kZWwgZGF0YQ==",
    "uploader": "api_user",
    "created_at": 1760824634662036200,
    "is_public": true,
    "prediction_accuracy": 0,
    "model_params": "{\"layers\": 5, \"neurons\": 128}"
  }
}
```

**Gdy brak modelu:**
```json
{
  "success": true,
  "data": null,
  "message": "No model is currently stored in the context"
}
```

---

### 4. POST `/model` - Upload nowego modelu ML

Uploaduje nowy model ML do kontekstu Calimero. **Zastępuje** poprzedni model.

```bash
curl -X POST http://localhost:3001/model \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My ML Model",
    "description": "Brain tumor classifier model",
    "model_type": "cnn_classifier",
    "version": "1.0.0",
    "file_bytes_base64": "SGVsbG8gV29ybGQh",
    "uploader": "john_doe",
    "prediction_accuracy": 0.95,
    "date": 1697654400,
    "model_params": "{\"layers\": 3, \"dropout\": 0.5}",
    "is_public": true
  }'
```

**Request Body:**

| Pole | Typ | Wymagane | Opis |
|------|-----|----------|------|
| `name` | string | ✅ | Nazwa modelu |
| `description` | string | ✅ | Opis modelu |
| `model_type` | string | ✅ | Typ modelu (np. "cnn_classifier", "neural_network") |
| `version` | string | ✅ | Wersja modelu (np. "1.0.0") |
| `file_bytes_base64` | string | ✅ | Dane modelu zakodowane w Base64 |
| `uploader` | string | ✅ | Nazwa użytkownika uploadującego |
| `prediction_accuracy` | number | ❌ | Dokładność predykcji (0-1), domyślnie 0 |
| `date` | number | ❌ | Timestamp, domyślnie aktualny czas |
| `model_params` | string | ❌ | Parametry modelu jako JSON string, domyślnie "{}" |
| `is_public` | boolean | ❌ | Czy model jest publiczny, domyślnie true |

**Odpowiedź sukcesu:**
```json
{
  "success": true,
  "data": {
    "model_id": "model:My ML Model:1.0.0",
    "message": "Model uploaded successfully"
  }
}
```

**Odpowiedź błędu (brak wymaganych pól):**
```json
{
  "success": false,
  "error": "Missing required fields: name, description, model_type, version, file_bytes_base64, uploader"
}
```

---

## 🔐 Uwagi dotyczące bezpieczeństwa

⚠️ **WAŻNE**: Ten serwer **NIE MA autoryzacji**. Endpointy są publicznie dostępne!

- Używaj tylko w środowisku deweloperskim
- NIE eksponuj na produkcji bez dodania autoryzacji
- Dodaj middleware do autentykacji (JWT, API keys, etc.) przed wdrożeniem

---

## 📝 Przykłady użycia

### Python

```python
import requests
import base64

# Pobierz obecny model
response = requests.get('http://localhost:3001/model')
model = response.json()
print(f"Current model: {model['data']['name']}")

# Upload nowego modelu
with open('model.keras', 'rb') as f:
    model_data = base64.b64encode(f.read()).decode('utf-8')

new_model = {
    "name": "Brain Tumor Classifier",
    "description": "CNN model for brain tumor classification",
    "model_type": "cnn_classifier",
    "version": "2.0.0",
    "file_bytes_base64": model_data,
    "uploader": "data_scientist",
    "prediction_accuracy": 0.99,
    "date": int(time.time()),
    "model_params": json.dumps({"layers": 5, "dropout": 0.3}),
    "is_public": True
}

response = requests.post('http://localhost:3001/model', json=new_model)
result = response.json()
print(f"Upload result: {result}")
```

### JavaScript/TypeScript

```javascript
// Pobierz obecny model
const getModel = async () => {
  const response = await fetch('http://localhost:3001/model');
  const data = await response.json();
  console.log('Current model:', data.data);
  return data.data;
};

// Upload nowego modelu
const uploadModel = async (modelFile) => {
  // Konwertuj plik do Base64
  const reader = new FileReader();
  const base64Data = await new Promise((resolve) => {
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(modelFile);
  });

  const newModel = {
    name: 'My Neural Network',
    description: 'Deep learning model',
    model_type: 'neural_network',
    version: '1.0.0',
    file_bytes_base64: base64Data,
    uploader: 'frontend_user',
    prediction_accuracy: 0.95,
    date: Date.now(),
    model_params: JSON.stringify({ layers: 3 }),
    is_public: true
  };

  const response = await fetch('http://localhost:3001/model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(newModel)
  });

  const result = await response.json();
  console.log('Upload result:', result);
  return result;
};
```

### cURL

```bash
# Health check
curl http://localhost:3001/health

# Pobierz model
curl http://localhost:3001/model

# Upload modelu (z pliku)
BASE64_DATA=$(base64 -w 0 model.keras)
curl -X POST http://localhost:3001/model \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"My Model\",
    \"description\": \"Test model\",
    \"model_type\": \"classifier\",
    \"version\": \"1.0.0\",
    \"file_bytes_base64\": \"$BASE64_DATA\",
    \"uploader\": \"user\",
    \"prediction_accuracy\": 0.95,
    \"date\": $(date +%s),
    \"model_params\": \"{}\",
    \"is_public\": true
  }"
```

---

## 🛠️ Konfiguracja

Konfiguracja znajduje się w pliku `.env`:

```env
VITE_NODE_URL=http://localhost:2528
VITE_CONTEXT_ID=DUYLik9nx5pvhXkshA9dCG4ya52mqHrM5TX3rJxw1LGS
VITE_APPLICATION_ID=HELDXwknx9tVnj3JKfa3EMyGB9JEsApeijVHzKn5cRVX
VITE_EXECUTOR_PUBLIC_KEY=DZn5moPKtaFMGETRGwezDKoxcoW2a4VR3L2GWvMJQByN
```

Port serwera można zmienić przez zmienną środowiskową:
```bash
PORT=4000 pnpm server
```

---

## 📊 Status Codes

| Code | Znaczenie |
|------|-----------|
| 200 | Sukces |
| 400 | Błąd w danych wejściowych (brak wymaganych pól) |
| 500 | Błąd serwera lub komunikacji z Calimero |

---

## 🔗 Linki

- **Serwer API**: http://localhost:3001
- **Calimero Node**: http://localhost:2528
- **Context ID**: DUYLik9nx5pvhXkshA9dCG4ya52mqHrM5TX3rJxw1LGS

