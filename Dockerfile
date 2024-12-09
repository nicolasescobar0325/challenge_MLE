FROM python:3.9-slim

WORKDIR /app

COPY Makefile Makefile
RUN make install 

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]