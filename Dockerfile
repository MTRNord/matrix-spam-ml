FROM rust:1.64

WORKDIR /app
COPY ./crates /app/crates
COPY ./Cargo.toml /app
COPY ./Cargo.lock /app
RUN cargo build --release
RUN ls -la target/release/build/tensorflow-sys-*/out
RUN find / -name libtensorflow*

RUN find . -type f -name libtensorflow.so.2 -exec cp {} /usr/lib/ \; \
    && find . -type f -name libtensorflow_framework.so.2 -exec cp {} /usr/lib/ \;
RUN ldconfig
RUN ldd /app/target/release/model_server

ENV MODEL_PATH /app/models/matrix_spam
# Copy the model files to the image
COPY ./models/spam_keras_1664583738.3207538 /app/models/matrix_spam

CMD ["/app/target/release/model_server"]
