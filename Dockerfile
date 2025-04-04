FROM python:3.12
WORKDIR /site
COPY . .

RUN ollama run deepseek-r1:7b
RUN pip install --upgrade pip & pip install -r requirements.txt
CMD ["python3", "api_server.py"]
