# Базовый образ с Ollama
FROM python:3.12-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg sudo && \
    rm -rf /var/lib/apt/lists/*

# Установка Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Копируем зависимости Python
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Настройка переменных окружения
ENV PATH="/root/.ollama/bin:$PATH"
ENV OLLAMA_HOST=0.0.0.0

# Экспонируем порты
EXPOSE 8080 11434

# Запускаем Ollama сервер и приложение
CMD ollama serve & sleep 2 && ollama pull deepseek-r1:7b && python3 api_server.py
