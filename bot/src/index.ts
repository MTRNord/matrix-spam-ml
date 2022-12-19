import {
    MatrixClient,
    SimpleFsStorageProvider,
    AutojoinRoomsMixin,
    // MessageEvent,
    // TextualMessageEventContent,
    RustSdkCryptoStorageProvider,
} from "matrix-bot-sdk";
import { readFile } from "fs/promises";
import { load } from "js-yaml";
import * as tf from "@tensorflow/tfjs-node";
import { Rank, Tensor } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";

type Config = {
    homeserver: string;
    accessToken: string;
    modelPath: string;
    // A private room for admins to see responses on reports and other secret activity.
    adminRoom: string;
    // A possibly public room where warnings land which also is used to issue actions from as an admin
    warningsRoom: string;
};

class Bot {
    public static async createBot() {
        const config = load(await readFile("./config.yaml", "utf8")) as Config;

        // Add some divider for clarity after tensorflow loaded up
        const line = '-'.repeat(process.stdout.columns);
        console.log(line);

        // Check if required fields are set
        if (config.homeserver == undefined && config.accessToken == undefined) {
            console.error("Missing homeserver and accessToken config values");
            process.exit(1);
        } else if (config.homeserver == undefined) {
            console.error("Missing homeserver config value");
            process.exit(1);
        } else if (config.accessToken == undefined) {
            console.error("Missing accessToken config value");
            process.exit(1);
        }

        if (config.adminRoom == undefined && config.warningsRoom == undefined) {
            console.error("Missing adminRoom and warningsRoom config values");
            process.exit(1);
        } else if (config.adminRoom == undefined) {
            console.error("Missing adminRoom config value");
            process.exit(1);
        } else if (config.warningsRoom == undefined) {
            console.error("Missing warningsRoom config value");
            process.exit(1);
        }

        if (config.adminRoom.startsWith("#")) {
            console.error("adminRoom config value needs to be a roomid starting with a \"!\"");
            process.exit(1);
        }
        if (config.warningsRoom.startsWith("#")) {
            console.error("warningsRoom config value needs to be a roomid starting with a \"!\"");
            process.exit(1);
        }


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
        const cryptoProvider = new RustSdkCryptoStorageProvider("./ml-bot-store");

        tf.enableProdMode()
        const model = await tf.node.loadSavedModel(config.modelPath);

        // Finally, let's create the client and set it to autojoin rooms. Autojoining is typical of bots to ensure
        // they can be easily added to any room.
        const client = new MatrixClient(homeserverUrl, accessToken, storage, cryptoProvider);

        // TODO replace with manual handling to need admin approval
        AutojoinRoomsMixin.setupOnClient(client);

        // Join rooms as needed but crash if missing
        await client.joinRoom(config.adminRoom);
        await client.joinRoom(config.warningsRoom);

        return new Bot(config, client, model);
    }

    private constructor(private config: Config, private client: MatrixClient, private model: TFSavedModel) {
        // Before we start the bot, register our command handler
        client.on("room.message", this.handleMessage.bind(this));

        // Now that everything is set up, start the bot. This will start the sync loop and run until killed.
        client.start().then(() => console.log("Bot started!"));
    }

    // This is the command handler we registered a few lines up
    private async handleMessage(roomId: string, event: any) {
        // Don't handle unhelpful events (ones that aren't text messages, are redacted, or sent by us)
        if (event['content']?.['msgtype'] !== 'm.text') return;
        if (event['sender'] === await this.client.getUserId()) return;

        const body = event['content']['body'];
        if (roomId !== this.config.adminRoom && roomId !== this.config.warningsRoom) {
            await this.checkSpam(event, body, roomId);
        }
    }

    private async checkSpam(event: any, body: string, roomId: string) {
        // Check if spam
        const data = tf.tensor([body])
        const prediction: Tensor<Rank.R2> = this.model.predict(data) as Tensor<Rank.R2>;
        const prediction_data: number[][] = await prediction.array() as number[][];
        console.log(`Prediction: ${prediction_data}`);

        const prediction_value = ((prediction_data[0] ?? [])[0] ?? 0);
        if (prediction_value > 0.8) {
            // TODO reverse resolve alias for the roomID and make pill.
            const alert_event_id = await this.client.sendHtmlText(this.config.warningsRoom, `<blockquote>\n<p>${body}</p>\n</blockquote>\n<p>Above message was detected as spam. See json file for full event and use reactions to take action or no action.</p>\n<p>It was sent in ${roomId}</p>\n<p>Spam Score is: ${prediction_value.toFixed(3)}</p>\n`);
            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, "🚨 Ban User");
            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, "⚠️ Kick User");
            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, "✅ False positive");
            var eventContent = Buffer.from(JSON.stringify(event), 'utf8');
            const media = await this.client.uploadContent(eventContent, "application/json", "event.json");
            await this.client.sendMessage(this.config.warningsRoom, {
                msgtype: "m.file",
                body: "event.json",
                filename: "event.json",
                info: {
                    mimetype: "application/json",
                    size: eventContent.length,
                },
                url: media,
            })
        } else {
            // const message = new MessageEvent(event);
            // const textEvent = new MessageEvent<TextualMessageEventContent>(message.raw);
            //await this.client.unstableApis.addReactionToEvent(roomId, textEvent.eventId, "Classified Not Spam")
        }
    }
}

await Bot.createBot();

