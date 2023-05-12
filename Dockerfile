FROM zamafhe/concrete-ml:latest

RUN useradd -u 4488 kron
USER kron

RUN mkdir /app_src
COPY ./requirements.txt /app_src/requirements.txt
RUN pip install --no-cache-dir -r /app_src/requirements.txt
COPY ./client /app_src/client/
COPY ./code_src /app_src/code_src/
COPY ./server /app_src/server

WORKDIR /app_src