# Matrix-Spam ML

This project consists of tooling to generate a spam detection model for the Matrix protocol.

It utilizes Tensorflow and builds the model in python and then provides a Rust server that provides some APIs to interact with the model easily and also extend it.

The current code base is fast moving. Expect to change rapidly.

## Usage

### Training

To train the model, you need to have a set of labeled data.
This data is at `./input/MatrixData`. It is a TSV file.
Please note that URLs should not be added as well as newlines.
Newlines will be stripped anyway and URLs tend to break the model result.

To train the model, run `python3 model_v2.py`. This will train the model and save it to `./model/`.
Please make sure you installed tensorflow.

#### Notes about the data

Please ensure to remove all urls, html tags and new lines. Also make sure to strip duplicate whitespace.
All of these reduce accuracy easily.

### Running the server

To run the server, run `cargo run --release`. This will start the server on port `3000`.

If you dont see any log try preprending `RUST_LOG=info` to the command.

## API

### POST /test

This endpoint takes a JSON body with the following format:

```json
{
    "input_data": "This is a message to be classified"
}
```

It will return a JSON response with the following format:

```json5
{
    "input_data": "This is a message to be classified",
    "score": 1.1349515e-24, // Note that this is a float
}
```

You do not have to strip urls like in the training data. However it might yield better results if you strip the html tags.

## Support

For support you can join [#matrix-spam-ml:midnightthoughts.space](https://matrix.to/#/#matrix-spam-ml:midnightthoughts.space) on Matrix.
