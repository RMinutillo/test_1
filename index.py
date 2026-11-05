# ---------------------------------------------------------------
# IMPORTS (todos juntos al principio del archivo)
# ---------------------------------------------------------------
import asyncio
import os
import sys
from enum import Enum
from typing import AsyncIterator

# python-dotenv: lee el archivo .env y carga sus valores como variables de entorno.
from dotenv import load_dotenv

# Importamos las herramientas que necesitamos de la librería Pydantic.
# - BaseModel: la "plantilla base" para crear clases que validan datos automáticamente.
# - Field: nos permite definir reglas para cada dato (mínimos, máximos, valor por defecto).
# - field_validator: nos permite escribir una regla de validación personalizada.
# - ValidationError: el error que lanza Pydantic cuando un dato no cumple las reglas.
from pydantic import BaseModel, Field, ValidationError, field_validator

# Carga las variables del archivo .env (si existe) en el entorno del programa.
# Debe ejecutarse ANTES de usar os.getenv(), por eso va arriba de todo.
load_dotenv()


# ---------------------------------------------------------------
# CONSTANTES DE CONFIGURACIÓN (se pueden ajustar)
# ---------------------------------------------------------------
MAX_ATTEMPTS = 3        # Cantidad total de intentos antes de rendirse
BASE_DELAY = 1.0        # Segundos de espera antes del 2do intento (luego 2, 4, 8...)
REQUEST_TIMEOUT = 30.0  # Segundos máximos de espera por la API antes de dar timeout


# ---------------------------------------------------------------
# PASO 1: VALIDACIÓN DE DATOS CON PYDANTIC
# ---------------------------------------------------------------

# Una "clase" es como un molde o formulario. Esta clase define qué datos necesitamos
# para llamar a la IA y qué reglas deben cumplir.
# Al heredar de BaseModel (lo que va entre paréntesis), la clase valida los datos
# sola: si algo no cumple las reglas, Python lanza un error.
class ChatConfig(BaseModel):
    """Parámetros de configuración para la llamada a la API de IA."""

    # --- CAMPO 1: el mensaje del usuario ---
    # "message: str" significa: el campo se llama "message" y debe ser texto (str = string).
    # Los tres puntos (...) significan que es OBLIGATORIO: no tiene valor por defecto.
    message: str = Field(..., description="Mensaje del usuario")

    # --- CAMPO 2: la temperatura ---
    # La temperatura controla qué tan "creativa" es la respuesta de la IA:
    # valores bajos = respuestas más predecibles, valores altos = más variadas.
    # "float" significa que es un número con decimales (por ejemplo 0.7).
    # NOTA: solo se usa con OpenAI. Con Anthropic se ignora.
    temperature: float = Field(
        default=0.7,  # Si el usuario no indica nada, se usa 0.7
        ge=0.0,       # ge = "greater or equal": debe ser MAYOR O IGUAL a 0.0
        le=2.0,       # le = "less or equal": debe ser MENOR O IGUAL a 2.0
        description="Creatividad de la respuesta (0.0 a 2.0)",
    )

    # --- CAMPO 3: cantidad máxima de tokens ---
    # Un "token" es un pedacito de texto (aprox. una palabra o parte de una palabra).
    # Este valor limita qué tan larga puede ser la respuesta de la IA.
    # "int" significa que es un número entero (sin decimales).
    max_tokens: int = Field(
        default=1000,  # Si no se indica nada, se usa 1000
        gt=0,          # gt = "greater than": debe ser MAYOR que 0 (el 0 no vale)
        le=4096,       # No puede superar 4096
        description="Cantidad máxima de tokens a generar",
    )

    # --- VALIDACIÓN PERSONALIZADA para el mensaje ---
    # Los símbolos @ se llaman "decoradores": agregan un comportamiento especial
    # a la función que está justo debajo.
    # Este decorador le dice a Pydantic: "ejecutá esta función para revisar el campo 'message'".
    @field_validator("message")
    @classmethod
    def message_no_vacio(cls, v: str) -> str:
        # "v" es el valor que escribió el usuario.
        # strip() borra los espacios al principio y al final del texto.
        # Ejemplo: "   hola   " se convierte en "hola".
        v = v.strip()

        # Si después de borrar los espacios no quedó nada, el mensaje estaba vacío.
        # "not v" es verdadero cuando el texto está vacío.
        if not v:
            # raise ValueError lanza un error con un mensaje explicativo.
            raise ValueError("El mensaje no puede estar vacío")

        # Si todo está bien, devolvemos el mensaje ya limpio.
        return v


# ---------------------------------------------------------------
# PASO 2: ERRORES PROPIOS DEL PROGRAMA
# ---------------------------------------------------------------
# OpenAI y Anthropic lanzan errores con nombres distintos. Para no tener que
# manejar los dos tipos por separado, los convertimos a estos errores propios.
# Así el resto del programa solo necesita conocer estos tres.

class LLMError(Exception):
    """Error general al hablar con la IA (clase base de los otros dos)."""


class TransientLLMError(LLMError):
    """Error TEMPORAL: tiene sentido volver a intentar.
    Ejemplos: servicio colapsado, rate limit, timeout, falla de conexión."""


class PermanentLLMError(LLMError):
    """Error que NO se arregla reintentando.
    Ejemplos: clave API inválida, modelo inexistente, pedido mal armado."""


# ---------------------------------------------------------------
# PASO 3: ADMINISTRADOR DE PROVEEDORES (OpenAI / Anthropic)
# ---------------------------------------------------------------

# Enum = lista cerrada de opciones válidas. Evita errores de tipeo como "opnai".
class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Modelo por defecto para cada proveedor (podés cambiarlos).
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",
    Provider.ANTHROPIC: "claude-sonnet-4-5",
}


class AsyncLLMManager:
    """Carga el cliente asíncrono de OpenAI o Anthropic según la configuración."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        # 1) Elegimos el proveedor: el que nos pasen, o el de la variable
        #    de entorno LLM_PROVIDER, o "openai" si no hay ninguno.
        provider_name = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

        # 2) Validamos que sea uno de los proveedores soportados.
        try:
            self.provider = Provider(provider_name)
        except ValueError:
            validos = ", ".join(p.value for p in Provider)
            raise ValueError(
                f"Proveedor '{provider_name}' no soportado. Opciones: {validos}"
            )

        # 3) Elegimos el modelo: el que nos pasen o el de por defecto.
        self.model = model or DEFAULT_MODELS[self.provider]

        # 4) Creamos el cliente correspondiente.
        self.client = self._load_client()

    def _load_client(self):
        """Crea el cliente async del proveedor elegido."""
        # max_retries=0 desactiva los reintentos automáticos de las librerías,
        # porque los reintentos los controlamos nosotros (así sabemos qué pasa).
        # timeout es el tiempo máximo de espera por la API.
        if self.provider == Provider.OPENAI:
            # Importamos acá adentro para que solo necesites tener instalada
            # la librería del proveedor que realmente usás.
            from openai import AsyncOpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Falta la variable de entorno OPENAI_API_KEY")
            return AsyncOpenAI(
                api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT
            )

        # Si no es OpenAI, es Anthropic.
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Falta la variable de entorno ANTHROPIC_API_KEY")
        return AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT
        )

    # -----------------------------------------------------------
    # PASO 4: TRADUCIR LOS ERRORES DE LAS LIBRERÍAS A LOS NUESTROS
    # -----------------------------------------------------------
    def _translate_error(self, exc: Exception) -> LLMError:
        """Convierte cualquier error en TransientLLMError o PermanentLLMError."""
        # Si ya es uno de los nuestros, lo dejamos como está.
        if isinstance(exc, LLMError):
            return exc

        # Las dos librerías ponen el código HTTP del error en "status_code".
        # (Ejemplo: 429 = demasiadas solicitudes, 503 = servicio no disponible.)
        # Si el error no vino de una respuesta HTTP, status vale None.
        status = getattr(exc, "status_code", None)
        name = type(exc).__name__

        # --- Errores TEMPORALES (vale la pena reintentar) ---
        if name == "APITimeoutError" or isinstance(exc, asyncio.TimeoutError):
            return TransientLLMError("La IA tardó demasiado en responder (timeout).")

        if name == "APIConnectionError" or isinstance(exc, ConnectionError):
            return TransientLLMError(
                "No se pudo conectar con la IA. Revisá tu conexión a internet."
            )

        if status == 429:
            return TransientLLMError(
                "Demasiadas solicitudes (rate limit). Si el error persiste, "
                "revisá que tu cuenta tenga crédito."
            )

        # Los códigos 500 o más indican problemas del lado del servidor
        # (incluye 503 = no disponible y 529 = sobrecargado en Anthropic).
        if status is not None and status >= 500:
            return TransientLLMError(
                f"El servicio de IA está caído o sobrecargado (código {status})."
            )

        if status in (408, 409):
            return TransientLLMError(f"Error temporal del servicio (código {status}).")

        # --- Errores PERMANENTES (reintentar no sirve) ---
        if status == 401:
            return PermanentLLMError(
                "Clave API inválida o mal copiada (revisá tu archivo .env)."
            )

        if status == 403:
            return PermanentLLMError("Tu clave API no tiene permisos para esta acción.")

        if status == 404:
            return PermanentLLMError(
                f"El modelo '{self.model}' no existe o no está disponible "
                "para tu cuenta."
            )

        if status == 400:
            return PermanentLLMError(f"La API rechazó el pedido (código 400): {exc}")

        if status is not None:
            return PermanentLLMError(f"Error de la API (código {status}): {exc}")

        # Cualquier otra cosa que no esperábamos.
        return PermanentLLMError(f"Error inesperado ({name}): {exc}")

    # -----------------------------------------------------------
    # PASO 5: UN INTENTO DE GENERACIÓN EN STREAMING (sin reintentos)
    # -----------------------------------------------------------
    async def _stream_once(self, config: ChatConfig) -> AsyncIterator[str]:
        """Hace UNA llamada a la API y entrega el texto en fragmentos con yield."""
        if self.provider == Provider.OPENAI:
            # stream=True le pide a OpenAI que mande la respuesta de a pedacitos.
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": config.message}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                stream=True,
            )
            # "async for" recorre los pedacitos a medida que van llegando.
            async for chunk in stream:
                # Algunos eventos vienen sin texto (por ejemplo el último),
                # por eso verificamos antes de entregar.
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        else:
            # Con Anthropic NO enviamos la temperatura: las versiones nuevas
            # del SDK ya no la aceptan como argumento. El modelo usa su valor
            # por defecto.
            # "async with" abre el stream y lo cierra solo al terminar.
            async with self.client.messages.stream(
                model=self.model,
                messages=[{"role": "user", "content": config.message}],
                max_tokens=config.max_tokens,
            ) as stream:
                # text_stream entrega solo el texto, sin metadatos.
                async for text in stream.text_stream:
                    yield text

    # -----------------------------------------------------------
    # PASO 6: GENERACIÓN EN STREAMING CON REINTENTOS
    # -----------------------------------------------------------
    async def generate_stream(self, config: ChatConfig) -> AsyncIterator[str]:
        """Genera la respuesta en fragmentos, reintentando si hay errores temporales.

        Reglas:
        - Error temporal ANTES de recibir texto  -> espera y reintenta.
        - Error temporal DESPUÉS de recibir texto -> NO reintenta, porque
          repetiría texto que el usuario ya vio. Avisa que se cortó.
        - Error permanente -> corta enseguida con un mensaje claro.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # "emitted" recuerda si ya entregamos algún fragmento en este intento.
            emitted = False
            try:
                async for fragment in self._stream_once(config):
                    emitted = True
                    yield fragment
                # Si llegamos acá, la respuesta terminó bien: salimos.
                return

            except Exception as exc:
                error = self._translate_error(exc)

                # Caso 1: se cortó a mitad de la respuesta.
                if emitted:
                    raise LLMError(
                        f"La respuesta se interrumpió a la mitad. Detalle: {error}"
                    ) from exc

                # Caso 2: error temporal y todavía quedan intentos -> reintentar.
                if isinstance(error, TransientLLMError) and attempt < MAX_ATTEMPTS:
                    # Espera creciente: 1 s, 2 s, 4 s...  (2 ** n es "2 a la n")
                    wait = BASE_DELAY * (2 ** (attempt - 1))
                    print(
                        f"[Aviso] {error} Reintentando en {wait:.0f} s "
                        f"(intento {attempt} de {MAX_ATTEMPTS})...",
                        file=sys.stderr,
                    )
                    await asyncio.sleep(wait)
                    continue  # Vuelve al principio del "for" para otro intento

                # Caso 3: error permanente, o se acabaron los intentos.
                if isinstance(error, TransientLLMError):
                    raise TransientLLMError(
                        f"{error} Se agotaron los {MAX_ATTEMPTS} intentos."
                    ) from exc
                raise error from exc


# ---------------------------------------------------------------
# PRUEBA: ejecuta todo junto
# ---------------------------------------------------------------
async def main():
    # --- Preparación: puede fallar por datos inválidos o falta de clave ---
    try:
        manager = AsyncLLMManager()
        config = ChatConfig(message="Explicame qué es Python en 3 oraciones")
    except ValidationError as e:
        # Los datos no cumplen las reglas de Pydantic (temperatura, tokens, etc.)
        print(f"Datos inválidos:\n{e}", file=sys.stderr)
        return
    except ValueError as e:
        # Falta una clave API o el proveedor no existe.
        print(f"Error de configuración: {e}", file=sys.stderr)
        return

    # --- Llamada a la IA: puede fallar por problemas del servicio ---
    try:
        # Recorremos los fragmentos y los imprimimos apenas llegan.
        # end="" evita el salto de línea y flush=True fuerza a mostrarlo al instante.
        async for fragmento in manager.generate_stream(config):
            print(fragmento, end="", flush=True)
        print()
    except TransientLLMError as e:
        print(f"\n[Error temporal] {e} Probá de nuevo en unos minutos.", file=sys.stderr)
    except LLMError as e:
        print(f"\n[Error] {e}", file=sys.stderr)


# Esta condición hace que main() solo corra si ejecutás este archivo directamente
# (y no si lo importás desde otro archivo).
if __name__ == "__main__":
    # asyncio.run arranca el programa asíncrono.
    asyncio.run(main())