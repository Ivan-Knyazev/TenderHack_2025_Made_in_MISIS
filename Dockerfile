FROM python:3.12
WORKDIR /site
COPY . .

CMD ["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"]
RUN ollama run deepseek-r1:7b
RUN pip install --upgrade pip & pip install -r requirements.txt
CMD ["python3", "api_server.py"]
