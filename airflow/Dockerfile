
FROM apache/airflow:3.4.0-python3.13

USER root

# Сначала устанавливаем which, необходимый системным скриптам
RUN apt-get update \
    && apt-get install -y --no-install-recommends which \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Java и сертификаты
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-21-jre-headless \
        ca-certificates \
        ca-certificates-java \
    && update-ca-certificates --fresh \
    && dpkg-reconfigure -f noninteractive ca-certificates-java \
    && keytool -list -cacerts -storepass changeit \
        | grep -q "trustedCertEntry" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow

COPY --chown=airflow:root requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.4.0" \
    -r /requirements.txt
