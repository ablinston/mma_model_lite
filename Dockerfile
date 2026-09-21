FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

# libtk8.6 is for pyautogui, which imports mouseinfo and dies without tkinter
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    openssh-client \
    curl \
    chromium \
    xvfb \
    xauth \
    libtk8.6 \
    && rm -rf /var/lib/apt/lists/*

RUN printf '%b' '[include]\n\tpath = /root/.gitconfig-host\n' \
                '[user]\n\tname = ablinston\n\temail = ablinston@users.noreply.github.com\n' \
                '[safe]\n\tdirectory = /workspace\n' > /root/.gitconfig

WORKDIR /workspace

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y



CMD ["bash"]
