# Chat con IA en streaming (OpenAI / Anthropic)

Programa en Python que recibe un mensaje, lo envía a la API de una IA (**OpenAI** o **Anthropic**, a elección) y muestra la respuesta **en streaming**, es decir, a medida que se va generando.

## Características

- **Validación de datos con Pydantic:** el mensaje, la temperatura y la cantidad de tokens se validan antes de llamar a la API.
- **Dos proveedores intercambiables:** se elige con una variable de configuración (`LLM_PROVIDER`), sin tocar el código.
- **Streaming asíncrono:** la respuesta se imprime fragmento a fragmento usando `async`/`yield`.
- **Manejo de errores:** distingue errores temporales (servicio colapsado, rate limit, timeout, falla de conexión) de errores permanentes (clave inválida, modelo inexistente) y reintenta automáticamente solo los temporales.
- **Variables de entorno con python-dotenv:** las claves API se guardan en un archivo `.env`, nunca en el código.

## Requisitos

- Python **3.10 o superior** (el código usa la sintaxis `str | None`).
- Una clave API de [Anthropic](https://console.anthropic.com) o de [OpenAI](https://platform.openai.com). La API se paga aparte de cualquier suscripción de chat (Claude Pro, ChatGPT Plus) y hay que cargar crédito en la cuenta.

## Estructura del proyecto

```
tu-carpeta/
├── index.py     # Todo el código
├── .env         # Tus claves y configuración (NO se sube a Git)
└── README.md
```

## Instalación

**1. Abrí una terminal en la carpeta del proyecto.**
En VS Code: menú Terminal → New Terminal.

**2. (Recomendado) Creá un entorno virtual:**

```bash
python -m venv venv
```

Activalo:

- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

**3. Instalá las dependencias:**

```bash
pip install pydantic openai anthropic python-dotenv
```

## Configuración

Creá un archivo llamado exactamente `.env` (con el punto al principio) en la misma carpeta que `index.py`:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=tu_clave_aqui
OPENAI_API_KEY=tu_clave_aqui
```

| Variable            | Descripción                                             |
| ------------------- | ------------------------------------------------------- |
| `LLM_PROVIDER`      | Proveedor a usar: `anthropic` u `openai`. Por defecto `openai`. |
| `ANTHROPIC_API_KEY` | Tu clave de Anthropic (solo si usás Anthropic).         |
| `OPENAI_API_KEY`    | Tu clave de OpenAI (solo si usás OpenAI).               |

Notas:

- Sin comillas ni espacios alrededor del `=`.
- Podés tener las dos claves cargadas y cambiar de proveedor modificando solo `LLM_PROVIDER`.
- En Windows, asegurate de que el archivo no quede guardado como `.env.txt`.
- Si usás Git, agregá `.env` a tu archivo `.gitignore` para no publicar tus claves por accidente.

## Cómo ejecutarlo

```bash
python index.py
```

En Mac/Linux, si `python` no funciona, usá `python3 index.py`.

**Resultado esperado:** el texto de la respuesta aparece de a pedacitos, como si se escribiera en vivo.

## Cómo usarlo con tu propio mensaje

El mensaje de ejemplo está en la función `main()`, al final de `index.py`:

```python
config = ChatConfig(message="Explicame qué es Python en 3 oraciones")
```

Cambiá el texto por tu pregunta. También podés ajustar los parámetros:

```python
config = ChatConfig(
    message="Escribí un poema corto sobre el mar",
    temperature=1.0,   # 0.0 a 2.0 (solo se aplica con OpenAI)
    max_tokens=500,    # 1 a 4096
)
```

Para elegir otro modelo o forzar un proveedor desde el código:

```python
manager = AsyncLLMManager(provider="openai", model="gpt-4o-mini")
```

### Reglas de validación

| Campo         | Regla                                              |
| ------------- | -------------------------------------------------- |
| `message`     | Texto no vacío (se eliminan espacios sobrantes).   |
| `temperature` | Número entre 0.0 y 2.0. Por defecto 0.7.           |
| `max_tokens`  | Entero entre 1 y 4096. Por defecto 1000.           |

Si algún valor no cumple, el programa muestra qué dato falló y no llama a la API.

> **Importante:** con Anthropic la temperatura **no se envía**. Las versiones nuevas del SDK de Anthropic ya no aceptan ese parámetro, así que el modelo usa su valor por defecto. Con OpenAI sí se aplica.

## Manejo de errores

Los errores de ambas librerías se convierten a dos tipos propios:

| Tipo        | Qué incluye                                                          | Qué hace el programa                 |
| ----------- | -------------------------------------------------------------------- | ------------------------------------ |
| Temporal    | Servicio caído o sobrecargado (500 o más), rate limit (429), timeout, falla de conexión | Reintenta con espera creciente       |
| Permanente  | Clave inválida (401), sin permisos (403), modelo inexistente (404), pedido rechazado (400) | Corta enseguida con un mensaje claro |

**Reintentos:** hasta 3 intentos en total, esperando 1 segundo antes del segundo y 2 segundos antes del tercero. Si se agotan, se informa el error.

**Regla del streaming:** solo se reintenta si el error ocurre *antes* de recibir el primer fragmento. Si la conexión se corta a mitad de la respuesta, no se reintenta (repetiría texto que ya se mostró) y se avisa que la respuesta quedó incompleta.

Podés ajustar el comportamiento con las constantes al principio de `index.py`:

```python
MAX_ATTEMPTS = 3        # Intentos totales
BASE_DELAY = 1.0        # Segundos de espera inicial (se duplica en cada intento)
REQUEST_TIMEOUT = 30.0  # Segundos máximos de espera por la API
```

### Cómo probar el manejo de errores

| Qué probar            | Cómo provocarlo                                              | Resultado esperado                                |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Clave inválida        | Poné una clave falsa en `.env`                               | Corta al instante: "Clave API inválida..."        |
| Modelo inexistente    | Poné un modelo que no existe en `DEFAULT_MODELS`             | Corta al instante: "El modelo ... no existe..."   |
| Falla de conexión     | Desconectá internet                                          | Avisos de reintento y luego el error final        |
| Datos inválidos       | Poné `temperature=5` en `ChatConfig`                         | Mensaje de validación de Pydantic                 |

## Cómo funciona (resumen del código)

1. **`ChatConfig`:** clase de Pydantic que valida mensaje, temperatura y tokens.
2. **`LLMError`, `TransientLLMError`, `PermanentLLMError`:** errores propios del programa.
3. **`AsyncLLMManager`:** elige y crea el cliente (OpenAI o Anthropic) según `LLM_PROVIDER`.
4. **`_translate_error`:** convierte los errores de las librerías en errores temporales o permanentes.
5. **`_stream_once`:** hace una llamada a la API y entrega el texto con `yield`.
6. **`generate_stream`:** envuelve lo anterior con la lógica de reintentos.
7. **`main`:** prepara la configuración, recorre los fragmentos con `async for` y captura los errores para mostrarlos de forma legible.

## Problemas comunes

| Síntoma                                         | Causa probable y solución                                                   |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| `Falta la variable de entorno ...`              | El `.env` no está en la misma carpeta, tiene otro nombre o la clave está vacía. |
| `ModuleNotFoundError`                           | Falta instalar una librería o el entorno virtual no está activado.          |
| `Clave API inválida`                            | Clave mal copiada o de otro proveedor.                                      |
| `El modelo ... no existe`                       | Cambiá el nombre en `DEFAULT_MODELS` por uno disponible en tu cuenta.       |
| `Demasiadas solicitudes (rate limit)` persistente | Revisá que tu cuenta de API tenga crédito.                                |
| `TypeError ... unexpected keyword argument 'temperature'` | Versión nueva del SDK de Anthropic. Esta versión del código ya lo evita. |