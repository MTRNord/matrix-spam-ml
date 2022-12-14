# Matrix-Spam ML

This project consists of tooling to generate a spam detection model for the Matrix protocol.

It utilizes Tensorflow and builds the model in python and then provides a Rust server that provides some APIs to interact with the model easily and also extend it.

The current code base is fast moving. Expect to change rapidly.

## Usage

### Training

To train the model, you need to have a set of labeled data.
This data is at `./input/MatrixData`. It is a TSV file.

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

## Ethical use/How the usage of the model is intended

The model is trained mainly on SMS and matrix spam.
It is not curently checked for racism or other discrimination factors against other groups.
It is also at this time not checked how exactly it reacts to various scenarios.
Therefor please keep this in mind while using the model.

Additionally the model was designed as a warning systems for admins and not as an automod.
This is important as a warning system has a lot wider tollerances while an automod should be very certain to take actions against people.

While following this paragraph isnt mandatory I hope you keep it in mind and use this model ethically and not for discrimination in the network.

## Future Plans

- [ ] Mjolnir plugin/patch for collecting spam and using it to retrain the model.
- [ ] Balancing the sample data
- [ ] Synapse Plugin to use as an automute bot across the whole HS.
- [ ] Mjolnir plugin that allows applying the suggestions

## Support

For support you can join [#matrix-spam-ml:midnightthoughts.space](https://matrix.to/#/#matrix-spam-ml:midnightthoughts.space) on Matrix.
