import {
    MatrixClient,
    SimpleFsStorageProvider,
    AutojoinRoomsMixin,
    RustSdkCryptoStorageProvider,
    LogService,
    RichConsoleLogger,
    MessageEvent,
    MessageEventContent,
    MatrixProfileInfo,
    CanonicalAliasEventContent,
    RoomNameEventContent,
} from "matrix-bot-sdk";
import { readFile } from "fs/promises";
import { load } from "js-yaml";
import * as tf from "@tensorflow/tfjs-node";
import { Rank, Tensor } from "@tensorflow/tfjs-node";
import { TFSavedModel } from "@tensorflow/tfjs-node/dist/saved_model";
import { ReactionEvent } from "./events/ReactionEvent";
import { htmlToText } from "html-to-text";
import { BAN_REACTION, FALSE_POSITIVE_REACTION, getReactionHandler, KICK_REACTION } from "./reactionHandlers/reactionHandler";
import BanListHandler from "./banlist/banlist";

export type Config = {
    homeserver: string;
    accessToken: string;
    modelPath: string;
    // A private room for admins to see responses on reports and other secret activity.
    adminRoom: string;
    // A possibly public room where warnings land which also is used to issue actions from as an admin
    warningsRoom: string;
    // TODO: Dont set this via config but via a setup in the room on first launch or via the cli or something.
    banlistRoom: string;
};

const THRESHOLD = 0.8;
const startUpMessage = "Bot is starting up...";

class Bot {
    private readonly policyRoomHandler = new BanListHandler(this.client, this.config);
    public static async createBot() {
        LogService.setLogger(new RichConsoleLogger());
        LogService.muteModule("Metrics");
        const config = load(await readFile("./config.yaml", "utf8")) as Config;

        // Add some divider for clarity after tensorflow loaded up
        const line = '-'.repeat(process.stdout.columns);
        console.log(line);

        // Check if required fields are set
        if (config.homeserver == undefined && config.accessToken == undefined) {
            LogService.error("index", "Missing homeserver and accessToken config values");
            process.exit(1);
        } else if (config.homeserver == undefined) {
            LogService.error("index", "Missing homeserver config value");
            process.exit(1);
        } else if (config.accessToken == undefined) {
            LogService.error("index", "Missing accessToken config value");
            process.exit(1);
        }

        if (config.adminRoom == undefined && config.warningsRoom == undefined) {
            LogService.error("index", "Missing adminRoom and warningsRoom config values");
            process.exit(1);
        } else if (config.adminRoom == undefined) {
            LogService.error("index", "Missing adminRoom config value");
            process.exit(1);
        } else if (config.warningsRoom == undefined) {
            LogService.error("index", "Missing warningsRoom config value");
            process.exit(1);
        }

        if (config.adminRoom.startsWith("#")) {
            LogService.error("index", "adminRoom config value needs to be a roomid starting with a \"!\"");
            process.exit(1);
        }
        if (config.warningsRoom.startsWith("#")) {
            LogService.error("index", "warningsRoom config value needs to be a roomid starting with a \"!\"");
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

        // TODO: replace with manual handling to need admin approval
        AutojoinRoomsMixin.setupOnClient(client);

        // Join rooms as needed but crash if missing
        await client.joinRoom(config.adminRoom);
        await client.joinRoom(config.warningsRoom);


        return new Bot(config, client, model);
    }

    private constructor(private config: Config, private client: MatrixClient, private model: TFSavedModel) {
        // Before we start the bot, register our command handler
        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        client.on("room.message", this.handleMessage.bind(this));
        // eslint-disable-next-line @typescript-eslint/no-misused-promises
        client.on("room.event", this.handleEvents.bind(this));

        // Now that everything is set up, start the bot. This will start the sync loop and run until killed.
        client.start().then(async () => {
            LogService.info("index", "Bot started!");
            // Send notice that bot is starting into both rooms
            await client.sendNotice(config.adminRoom, startUpMessage);
            await client.sendNotice(config.warningsRoom, startUpMessage);
        }).catch(console.error);
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private async handleEvents(roomId: string, ev: any): Promise<void> {
        // For now only handle reactions
        const event = new ReactionEvent(ev);
        if (event.isRedacted) return; // Ignore redacted events
        if (event.sender === await this.client.getUserId()) return; // Ignore ourselves


        try {
            await getReactionHandler(roomId, event, this.client, this.config, this.policyRoomHandler).handleReaction();
        } catch (e) {
            return;
            // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
            //LogService.error("index", `Error handling reaction: ${e}`);
        }
    }

    // This is the command handler we registered a few lines up
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private async handleMessage(roomId: string, ev: any): Promise<void> {
        const event = new MessageEvent(ev);
        if (event.isRedacted) return; // Ignore redacted events that come through
        if (event.sender === await this.client.getUserId()) return; // Ignore ourselves
        if (event.messageType !== "m.text") return; // Ignore non-text messages

        if (roomId !== this.config.adminRoom && roomId !== this.config.warningsRoom) {
            await this.checkSpam(event, event.content?.body.trim() ?? "", roomId);
        }
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    private async checkSpam(event: MessageEvent<MessageEventContent>, body: string, roomId: string) {
        // Check if spam
        const data = tf.tensor([body])
        const prediction: Tensor<Rank.R2> = this.model.predict(data) as Tensor<Rank.R2>;
        const prediction_data: number[][] = await prediction.array();
        LogService.info("index", `Checking: "${body}"`);
        LogService.info("index", `Prediction: ${prediction_data.toString()}`);

        const prediction_value = ((prediction_data[0] ?? [])[0] ?? 0);
        if (prediction_value > THRESHOLD) {
            const mxid = event.sender;
            const displayname = (await (this.client.getUserProfile(mxid) as Promise<MatrixProfileInfo>)).displayname ?? mxid;
            let alias = roomId;
            let roomname = roomId;
            let room_url = `matrix:roomid/${roomId.replace("!", "")}?via=${this.config.homeserver.replace("https://", "")}`;
            try {
                alias = (await (this.client.getRoomStateEvent(roomId, "m.room.canonical_alias", "") as Promise<CanonicalAliasEventContent>)).alias;
                roomname = alias;
                room_url = `matrix:r/${alias.replace("#", "").replace("!", "")}`;
            } catch (e) {
                LogService.debug("index", `Failed to get alias for ${roomId}`);
            }
            try {
                roomname = (await (this.client.getRoomStateEvent(roomId, "m.room.name", "") as Promise<RoomNameEventContent>)).name;
            } catch (e) {
                LogService.debug("index", `Failed to get name for ${roomId}`);
            }

            const html = `<blockquote>\n<p>${body}</p>\n</blockquote>\n<p>Above message was detected as spam. See json file for full event and use reactions to take action or no action.</p><p>It was sent by <a href="matrix:u/${mxid.replace("@", "")}">${displayname}</a> in <a href="${room_url}">${roomname}</a></p><p>Spam Score is: ${prediction_value.toFixed(3)}</p>\n`;
            const roominfo: { roomId: string; name: string | undefined; alias: string | undefined; } = {
                roomId: roomId,
                name: undefined,
                alias: undefined
            };
            if (roomname !== roomId) {
                roominfo["name"] = roomname;
            }
            if (alias !== roomId) {
                roominfo["alias"] = alias;
            }
            const alert_event_id = await this.client.sendMessage(this.config.warningsRoom, {
                body: htmlToText(html, { wordwrap: false }),
                msgtype: "m.text",
                format: "org.matrix.custom.html",
                formatted_body: html,
                "space.midnightthoughts.spam_score": prediction_value.toFixed(3),
                "space.midnightthoughts.sending_user": mxid,
                "space.midnightthoughts.sending_room": roominfo,
                "space.midnightthoughts.event_id": event.eventId,
            });

            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, BAN_REACTION);
            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, KICK_REACTION);
            await this.client.unstableApis.addReactionToEvent(this.config.warningsRoom, alert_event_id, FALSE_POSITIVE_REACTION);
            const eventContent = Buffer.from(JSON.stringify(event.raw), 'utf8');
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
            // const textEvent = new MessageEvent<TextualMessageEventContent>(event.raw);
            //await this.client.unstableApis.addReactionToEvent(roomId, textEvent.eventId, "Classified Not Spam")
        }
    }
}

await Bot.createBot();
