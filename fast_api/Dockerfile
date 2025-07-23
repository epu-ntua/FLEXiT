FROM python:3.9.16-slim

RUN pip3 install --upgrade pip

COPY ./apisrc /app/src
COPY ./requirements_api.txt /app

WORKDIR /app

RUN pip3 install --default-timeout=1000 -r requirements_api.txt

EXPOSE 8000

#NO SSL
CMD ["uvicorn", "src.service:app", "--host=0.0.0.0", "--reload"]

