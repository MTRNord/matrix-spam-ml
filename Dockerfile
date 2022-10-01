FROM rust:1.64 as builder

WORKDIR /app
COPY ./crates /app/crates
COPY ./Cargo.toml /app
COPY ./Cargo.lock /app
RUN cargo build --release

ENV MODEL_PATH /app/models/matrix_spam
# Copy the model files to the image
COPY ./models/spam_keras_1664583738.3207538 /app/models/matrix_spam

CMD ["./target/release/model_server"]