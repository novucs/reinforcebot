# ReinforceBot Web Services

## Prerequisites
* [docker](https://docs.docker.com/install/)
* [docker-compose](https://docs.docker.com/compose/install/)
* A web browser

## QuickStart

```bash
docker-compose up -d
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py collectstatic --no-input
```

Open in browser: http://localhost:8080/
