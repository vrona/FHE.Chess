"""version: "3"

services:
  
  fhe_chess:
    build: .
    container_name: fhechess-s
    command: python3 --host 0.0.0.0 --port 80 --reload
    ports:
      - 80:80
    volumes:
      - shared-volume:/app_src
  
  server:
    build: server/
    command: python3 ./server.py
    ports:
      - 5555:5555
    volumes:
      - shared-volume:/app_src

  client:
    build: client/
    command: python3 ./chess_env/main.py
    network_mode: host
    depends_on:
      - server
    volumes:
      - shared-volume:/app_src

volumes:
  shared-volume:

  """