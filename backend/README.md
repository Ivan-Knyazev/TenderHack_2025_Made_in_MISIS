# FastAPI backend application with MongoDB for TenderHack

## Application functionality (endpoints)

* `/api/v1/query/generate` - for generate answer to the question in ML and put in to MongoDB (<b>main endpoint</b> of the system)
* `/api/v1/query/rate` - rate answer
* `/api/v1/query/analitycs` - get analitycs (for 2 charts)
* `/api/v1/query/all` - for get all queries with answers
<br>

* `/api/v1/auth/register` - for registration new admins
* `/api/v1/auth/login` - for logins admins
<br>

* `/api/v1/users/me` - get info about admin (protected)
<br>

* `/files/{file_path}` - get file (source of data to ML answer)


## Run application

### 1. Start MongoDB

Run with Docker (from `/backend` folder)

```
make db-up
```

### 2. Start FastAPI application (local usage)

- Start from `/backend` folder

```
uvicorn app.main:app --reload --port 8001 --host 0.0.0.0
```

### 3. See Swagger

Go to `http://localhost:8001/docs` in browser


## Some specifications for REST API:

- Change `backendhost` to really host!

For `registration` new user:
```
curl -X 'POST' \
  'http://backendhost:8000/api/v1/auth/register' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "username": "test@example.com",
  "full_name": "John Doe",
  "is_active": true,
  "is_admin": true,
  "hashed_password": "a5bd1f14adcda15f127a87b8ebc5a705232ec2c9f83d6447b5f958f7d4580a56"
}'
```

For `login` - get JWT Token:
```
curl -X 'POST' \
  'http://backendhost:8000/api/v1/auth/login' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "username": "test@example.com",
  "hashed_password": "a5bd1f14adcda15f127a87b8ebc5a705232ec2c9f83d6447b5f958f7d4580a56"
}'
```

For get `profile`:
```
curl -X 'GET' \
  'http://localhost:8000/api/v1/users/me?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzQzODM2NDg2LCJpYXQiOjE3NDM4MzU1ODZ9.ehEMuG3qc8thh3spjgbwMEHSkrO08hEeugV10Pv_1cw' \
  -H 'accept: application/json'
```

<hr/>

## Settings for MongoDB:

For connect to mongo from linux `bash`:
```bash
mongosh --host localhost --port 27017 -u mongo -p pwd --authenticationDatabase admin
```

In `mongosh`:
```
use aiassistant
db.createCollection("users")
db.createCollection("requests")
```