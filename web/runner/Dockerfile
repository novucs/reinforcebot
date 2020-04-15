FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev musl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
COPY . .
EXPOSE 8000
ENTRYPOINT ["gunicorn", "runner.wsgi"]
