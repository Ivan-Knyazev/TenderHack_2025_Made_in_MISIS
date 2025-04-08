FROM python:3.12.3
WORKDIR /app
COPY . .


RUN pip install --upgrade pip & pip install -r requirements.txt
CMD ["uvicorn", "app.main:app", "--reload" ,"--port", "8001", "--host", "0.0.0.0"]