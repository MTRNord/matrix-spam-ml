import {
    MatrixClient,
    SimpleFsStorageProvider,
    AutojoinRoomsMixin,
    MessageEvent,
    TextualMessageEventContent,
    //RustSdkCryptoStorageProvider,
} from "matrix-bot-sdk";
import { readFile } from "fs/promises";
import { load } from "js-yaml";
import * as tf from "@tensorflow/tfjs-node";
import { Rank, Tensor } from "@tensorflow/tfjs-node";

type Config = {
    homeserver: string;
    accessToken: string;
    modelPath: string;
};

const config = load(await readFile("./config.yaml", "utf8")) as Config;

// This will be the URL where clients can reach your homeserver. Note that this might be different
// from where the web/chat interface is hosted. The server must support password registration without
// captcha or terms of service (public servers typically won't work).
const homeserverUrl = config.homeserver;

// Use the access token you got from login or registration above.
const accessToken = config.accessToken;

// In order to make sure the bot doesn't lose its state between restarts, we'll give it a place to cache
// any information it needs to. You can implement your own storage provider if you like, but a JSON file
// will work fine for this example.
const storage = new SimpleFsStorageProvider("ml-bot.json");
// Broken
//const cryptoProvider = new RustSdkCryptoStorageProvider("./ml-bot-store");

const model = await tf.node.loadSavedModel(config.modelPath);

// Finally, let's create the client and set it to autojoin rooms. Autojoining is typical of bots to ensure
// they can be easily added to any room.
//const client = new MatrixClient(homeserverUrl, accessToken, storage, cryptoProvider);
const client = new MatrixClient(homeserverUrl, accessToken, storage);
AutojoinRoomsMixin.setupOnClient(client);

// Before we start the bot, register our command handler
client.on("room.message", handleMessage);

// Now that everything is set up, start the bot. This will start the sync loop and run until killed.
client.start().then(() => console.log("Bot started!"));

// This is the command handler we registered a few lines up
async function handleMessage(roomId: string, event: any) {
    // Don't handle unhelpful events (ones that aren't text messages, are redacted, or sent by us)
    if (event['content']?.['msgtype'] !== 'm.text') return;
    if (event['sender'] === await client.getUserId()) return;

    const body = event['content']['body'];
    console.log(`Checking: "${body}"`)

    // Check if spam
    const data = tf.tensor([body])
    const prediction: Tensor<Rank> = model.predict(data) as Tensor<Rank>;
    const prediction_data: number[][] = await prediction.array() as number[][];
    console.log(`Prediction: ${prediction_data}`);


    const message = new MessageEvent(event);
    const textEvent = new MessageEvent<TextualMessageEventContent>(message.raw);
    if (((prediction_data[0] ?? [])[0] ?? 0) > 0.8) {
        await client.unstableApis.addReactionToEvent(roomId, textEvent.eventId, "Classified Spam")
    } else {
        //await client.unstableApis.addReactionToEvent(roomId, textEvent.eventId, "Classified Not Spam")
    }


}
