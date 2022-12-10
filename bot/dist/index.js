import { MatrixClient, SimpleFsStorageProvider, AutojoinRoomsMixin, MessageEvent, } from "matrix-bot-sdk";
import { readFile } from "fs/promises";
import { load } from "js-yaml";
import * as tf from "@tensorflow/tfjs-node";
const config = load(await readFile("./config.yaml", "utf8"));
const homeserverUrl = config.homeserver;
const accessToken = config.accessToken;
const storage = new SimpleFsStorageProvider("ml-bot.json");
const model = await tf.node.loadSavedModel(config.modelPath);
const client = new MatrixClient(homeserverUrl, accessToken, storage);
AutojoinRoomsMixin.setupOnClient(client);
client.on("room.message", handleMessage);
client.start().then(() => console.log("Bot started!"));
async function handleMessage(roomId, event) {
    if (event['content']?.['msgtype'] !== 'm.text')
        return;
    if (event['sender'] === await client.getUserId())
        return;
    const body = event['content']['body'];
    console.log(`Checking: "${body}"`);
    const data = tf.tensor([body]);
    const prediction = model.predict(data);
    const prediction_data = await prediction.array();
    console.log(`Prediction: ${prediction_data}`);
    const message = new MessageEvent(event);
    const textEvent = new MessageEvent(message.raw);
    if (((prediction_data[0] ?? [])[0] ?? 0) > 0.8) {
        await client.unstableApis.addReactionToEvent(roomId, textEvent.eventId, "Classified Spam");
    }
    else {
    }
}
//# sourceMappingURL=index.js.map