FROM python:3.13-slim

# Evitar escritura de bytes y logs bufferizados por defecto
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar uv desde la imagen oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copiar configuración de proyecto
COPY pyproject.toml uv.lock ./

# Instalar estrictamente las librerias excluyendo el proyecto propio (evita error si no encuentra paquete source)
RUN uv sync --frozen --no-dev --no-install-project

# Exponer el binario del entorno virtual al PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copiar el resto del repositorio
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
