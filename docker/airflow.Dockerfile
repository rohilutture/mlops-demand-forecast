FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
USER airflow
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# requirements.txt upgrades `cryptography` past what Airflow's bundled pyOpenSSL supports
# (webserver crashes with "module 'lib' has no attribute 'GEN_EMAIL'"), so upgrade pyOpenSSL to match.
RUN pip install --no-cache-dir -U pyopenssl
