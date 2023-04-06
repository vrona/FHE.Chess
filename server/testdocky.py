"""FROM zamafhe/concrete-ml:v0.4.0

WORKDIR /root_server
COPY requirements.txt /root_server/requirements.txt

RUN pip install --no-cache-dir --upgrade /root_server/requirements.txt
ADD server /root_server

CMD [ "python3" ]"""