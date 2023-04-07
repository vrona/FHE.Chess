FROM zamafhe/concrete-ml:v0.4.0

WORKDIR /fhe_chess
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./client ./client
COPY ./code_src ./code_src
COPY ./server ./server

CMD ["python3"]