FROM python:3.8.1-alpine
RUN apk update \
    && apk add gcc postgresql-dev musl-dev \
    && rm -rf /var/cache/apk/*
WORKDIR /usr/src/app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
COPY . .
EXPOSE 8000
ENTRYPOINT ["gunicorn", "runner.wsgi"]
