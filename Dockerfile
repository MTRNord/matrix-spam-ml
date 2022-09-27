FROM rust:1.64 as builder

WORKDIR /app
COPY ./crates /app
COPY ./Cargo.toml /app
COPY ./Cargo.lock /app
RUN cargo build --release

ENV MODEL_PATH /app/models/matrix_spam
# Copy the model files to the image
COPY ./models/spam_keras_1664303305.1441052 /app/models/matrix_spam

CMD ["./target/release/model_server"]