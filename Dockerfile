FROM rust:1.64 as builder

# Install tensorflow
RUN apt-get update && apt-get install -y curl

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which python3) /usr/local/bin/python

RUN python3 -m pip install --no-cache-dir tensorflow

WORKDIR /app
COPY ./crates /app/crates
COPY ./Cargo.toml /app
COPY ./Cargo.lock /app
RUN cargo build --release
RUN ls -la target/release/build/tensorflow-sys-*/out
RUN find / -name libtensorflow*
RUN ldd /app/target/release/model_server

ENV MODEL_PATH /app/models/matrix_spam
# Copy the model files to the image
COPY ./models/spam_keras_1664583738.3207538 /app/models/matrix_spam

CMD ["/app/target/release/model_server"]
