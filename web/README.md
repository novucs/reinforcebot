# ReinforceBot Web Services

## Prerequisites
* [docker](https://docs.docker.com/install/)
* [docker-compose](https://docs.docker.com/compose/install/)
* [stripe-cli](https://stripe.com/docs/stripe-cli)
* A web browser

## QuickStart

```bash
stripe listen \
    --api-key sk_test_M1kmWdum2W1EoRSgBjrUTA5N00bySeWb1K \
    --events payment_intent.succeeded \
    --forward-to=http://localhost:8080/api/payments/
```

Copy Stripe webhook secret into `./django/.env.dev -> STRIPE_WEBHOOK_SECRET`

```
docker-compose up -d
docker-compose exec web python manage.py migrate
docker-compose exec web python manage.py collectstatic --no-input
```

Open in browser: http://localhost:8080/
