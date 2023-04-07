FROM zamafhe/concrete-ml:v0.4.0

RUN mkdir /app_src
COPY ./requirements.txt /app_src/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./client /app_src/client/
COPY ./code_src /app_src/code_src/
COPY ./server /app_src/server

WORKDIR /app_src
CMD ["python3", "--host", "0.0.0.0", "--port", "80"]