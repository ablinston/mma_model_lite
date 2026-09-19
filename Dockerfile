FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    git \
    ca-certificates \
    openssh-client \
    curl \
    chromium \
    && rm -rf /var/lib/apt/lists/*

RUN printf '%b' '[include]\n\tpath = /root/.gitconfig-host\n' \
                '[user]\n\tname = ablinston\n\temail = ablinston@users.noreply.github.com\n' \
                '[safe]\n\tdirectory = /workspace\n' > /root/.gitconfig

WORKDIR /workspace

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential python3.11-dev \
    && apt-get autoremove -y



CMD ["bash"]
