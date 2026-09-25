# python:3.11-slim over -bullseye/-alpine: alpine's musl libc breaks manylinux
# wheels (torch, scipy) forcing slow from-source builds; slim keeps glibc
# compatibility with prebuilt wheels while still being far smaller than the
# full python:3.11 image (~150MB base vs ~1GB). torch's CPU wheel itself is
# the biggest chunk of the final image (~600MB+) -- there is no way around
# that for a PyTorch service short of not shipping torch.
FROM python:3.11-slim

WORKDIR /app

# Copy dependency files first so Docker's layer cache is reused on every
# rebuild that only changes source code, not dependencies.
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps

# Non-root: never run a network-facing service as root.
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# No `curl` in slim images by default (installing it is another apt layer
# just for a healthcheck) -- Python's stdlib can hit the endpoint just as
# well with zero extra dependencies.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

# MODEL_DIR must be provided at `docker run` time (env var + a volume mount
# of an artifact produced by `python -m anomaly.export`) -- no model is
# baked into the image, so the same image serves any exported artifact.
CMD ["uvicorn", "anomaly.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
