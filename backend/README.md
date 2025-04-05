# Backend

## Specification for REST API:

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