# BHDAM-R PoC — reproducible container
# zfec is a C extension; build-essential lets the image build on ARM/Apple
# Silicon too (on Linux x86_64 pip uses a prebuilt wheel). Pinned deps.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole build context (respecting .dockerignore) so the build does not
# depend on the exact set/order of source filenames.
COPY . .

# Fail early and clearly if the main script did not make it into the context.
RUN test -f experiments.py \
    || (echo "ERROR: experiments.py not found in build context. Make sure the .py files, Dockerfile and requirements.txt are all in the folder you run 'docker build' from." && exit 1)

VOLUME ["/app/results"]
ENV MPLBACKEND=Agg

ENTRYPOINT ["python", "experiments.py", "--out", "/app/results"]
CMD ["all"]
