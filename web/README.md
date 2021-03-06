# ReinforceBot Web Services

## Prerequisites
* [docker](https://docs.docker.com/install/)
* [docker-compose](https://docs.docker.com/compose/install/)
* [stripe-cli](https://stripe.com/docs/stripe-cli)
* A web browser

## QuickStart
Start stripe listener.

```bash
stripe listen \
    --api-key sk_test_M1kmWdum2W1EoRSgBjrUTA5N00bySeWb1K \
    --events payment_intent.succeeded \
    --forward-to=http://localhost:8080/api/payments/
```

Copy Stripe webhook secret into `./api/.env.dev -> STRIPE_WEBHOOK_SECRET`

Gzip a tarball of the desktop client into the `blobs` directory. Desktop client
build instructions can be found in `../client/README.md`.

```bash
tar -czvf blobs/reinforcebot-client.tar.gz -C ../client/dist reinforcebot
```

Start and initialise all containers.

```bash
docker-compose up -d
docker-compose exec api python manage.py migrate
docker-compose exec api python manage.py collectstatic --no-input
```

Open in browser: http://localhost:8080/

PyCharm users: Mark `runner` and `api` directories as source roots.
