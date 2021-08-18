FROM python:3.7
WORKDIR /app

COPY static /app/static
COPY templates /app/templates
COPY transformers /app/transformers
COPY app.py /app
COPY requirements.txt /app
COPY california_housing.pkl /app

RUN pip install -r ./requirements.txt

EXPOSE 5000

# CMD ["gunicorn", "--preload", "--bind=:5000", "--forwarded-allow-ips=localhost", "app:app"]

ENTRYPOINT [ "flask" ]
CMD ["run", "--host=0.0.0.0", "--port=5000"]