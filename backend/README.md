# Backend

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