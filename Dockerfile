FROM python3.11.11-slim-buster

WORKDIR /squatbuddymodel

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "run.py"]