FROM apache/airflow:2.8.3
COPY airflow_env/requirements.txt /home/airflow/
RUN pip install --no-cache-dir -r /home/airflow/requirements.txt
COPY airflow_env /home/airflow/

USER root
RUN apt update && \
    apt install -y --no-install-recommends git && \
    apt install -y --no-install-recommends libgomp1 && \
    apt autoremove -yqq --purge && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
RUN apt install /home/airflow/dvc.deb
USER airflow
