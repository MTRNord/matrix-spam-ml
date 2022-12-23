import { LogService, MatrixClient } from "matrix-bot-sdk";
import BanListHandler from "../banlist/banlist";
import { ReactionEvent, ReactionEventContent } from "../events/ReactionEvent";
import { Config } from "../index";

export const BAN_REACTION = "🚨 Ban User";
export const KICK_REACTION = "⚠️ Kick User";
export const FALSE_POSITIVE_REACTION = "✅ False positive";


export function getReactionHandler<T extends ReactionEventContent>(roomId: string, event: ReactionEvent<T>, client: MatrixClient, config: Config, policyRoomHandler: BanListHandler): ReactionHandler<T> {
    switch (event.reaction) {
        case BAN_REACTION:
            return new BanReactionHandler(roomId, event, client, config, policyRoomHandler);
        case KICK_REACTION:
            return new KickReactionHandler(roomId, event, client, config, policyRoomHandler);
        case FALSE_POSITIVE_REACTION:
            return new FalsePositiveReactionHandler(roomId, event, client, config, policyRoomHandler);
        default:
            throw new Error("Reaction not supported");
    }
}

export abstract class ReactionHandler<T extends ReactionEventContent> {
    constructor(protected readonly roomId: string, protected readonly event: ReactionEvent<T>, protected readonly client: MatrixClient, protected readonly config: Config, protected readonly policyRoomHandler: BanListHandler) { }

    public abstract handleReaction(): Promise<void>;
}


class BanReactionHandler<T extends ReactionEventContent> extends ReactionHandler<T> {
    public async handleReaction(): Promise<void> {
        // Check if reaction was issued by an admin that can do this action
        // Get users in the admin room
        const users = await this.client.getJoinedRoomMembers(this.config.adminRoom);

        // Get power levels in the admin room
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const powerLevels = await this.client.getRoomStateEvent(this.config.adminRoom, "m.room.power_levels", "");

        // Map users to their power level
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        const admins = users.filter(u => powerLevels.content.users[u] >= powerLevels.content.ban);

        // Check if sender is an admin
        if (!admins.includes(this.event.sender)) {
            return;
        }



        let reactedToEvent: { content: { "space.midnightthoughts.sending_user": string; "space.midnightthoughts.previous_action": string; } } | undefined;
        try {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            reactedToEvent = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");

        } catch (e) {
            LogService.error("index", "Could not get event", e);
            return;
        }

        // Making typescript happy
        if (reactedToEvent == undefined) {
            LogService.error("index", "Could not get event");
            return;
        }

        if (process.env["DEBUG"] == "true") {
            LogService.info("BanReactionHandler", `Admin selected ban for ${reactedToEvent.content["space.midnightthoughts.sending_user"]} on ${this.event.targetEventId ?? "unknown"}`);
            return;
        }

        // We dont need to issue this twice for the same user
        // TODO: check if the user is already banned
        if (reactedToEvent.content["space.midnightthoughts.previous_action"] !== "ban") {
            await this.policyRoomHandler.banUser(reactedToEvent.content["space.midnightthoughts.sending_user"], "Banned by admin");

            // Get the message we sent in the warning room to update it
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            const warningMessage = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");
            // Update the event to reflect the action
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            warningMessage.content["space.midnightthoughts.previous_action"] = "ban";
            // Add relates_to to the event
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            warningMessage.content["m.relates_to"] = {
                "rel_type": "m.replace",
                // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                "event_id": this.event.targetEventId ?? ""
            };
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            warningMessage.content["m.new_content"] = {
                "msgtype": "m.text",
                // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
                "body": warningMessage.content.body,
                "format": "org.matrix.custom.html",
                // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
                "formatted_body": warningMessage.content.formatted_body
            }
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            await this.client.sendEvent(this.roomId, "m.room.message", warningMessage.content)
        }
    }
}

class KickReactionHandler<T extends ReactionEventContent> extends ReactionHandler<T> {
    public async handleReaction(): Promise<void> {
        // Check if reaction was issued by an admin that can do this action
        // Get users in the admin room
        const users = await this.client.getJoinedRoomMembers(this.config.adminRoom);

        // Get power levels in the admin room
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const powerLevels = await this.client.getRoomStateEvent(this.config.adminRoom, "m.room.power_levels", "");

        // Map users to their power level
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
        const admins = users.filter(u => powerLevels.content.users[u] >= powerLevels.content.kick);

        // Check if sender is an admin
        if (!admins.includes(this.event.sender)) {
            return;
        }

        let reactedToEvent: { content: { "space.midnightthoughts.sending_user": string; "space.midnightthoughts.previous_action": string; } } | undefined;
        try {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            reactedToEvent = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");

        } catch (e) {
            LogService.error("index", "Could not get event", e);
            return;
        }

        // Making typescript happy
        if (reactedToEvent == undefined) {
            LogService.error("index", "Could not get event");
            return;
        }

        if (process.env["DEBUG"]?.toLowerCase() == "true" || process.env["DEBUG"] == "1") {
            LogService.info("BanReactionHandler", `Admin selected kick for ${reactedToEvent.content["space.midnightthoughts.sending_user"]} on ${this.event.targetEventId ?? "unknown"}`);
            return;
        }

        await this.policyRoomHandler.kickUser(reactedToEvent.content["space.midnightthoughts.sending_user"], "Kicked by admin");
        // Get the message we sent in the warning room to update it
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const warningMessage = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");
        // Update the event to reflect the action
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        warningMessage.content["space.midnightthoughts.previous_action"] = "kick";
        // Add relates_to to the event
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        warningMessage.content["m.relates_to"] = {
            "rel_type": "m.replace",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            "event_id": this.event.targetEventId ?? ""
        };
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        warningMessage.content["m.new_content"] = {
            "msgtype": "m.text",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
            "body": warningMessage.content.body,
            "format": "org.matrix.custom.html",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
            "formatted_body": warningMessage.content.formatted_body
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        await this.client.sendEvent(this.roomId, "m.room.message", warningMessage.content)
    }
}

class FalsePositiveReactionHandler<T extends ReactionEventContent> extends ReactionHandler<T> {
    public async handleReaction(): Promise<void> {
        let reactedToEvent: { content: { "space.midnightthoughts.sending_user": string; "space.midnightthoughts.previous_action": string; } } | undefined;
        try {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
            reactedToEvent = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");

        } catch (e) {
            LogService.error("index", "Could not get event", e);
            return;
        }

        // Making typescript happy
        if (reactedToEvent == undefined) {
            LogService.error("index", "Could not get event");
            return;
        }

        LogService.info("BanReactionHandler", `Admin selected false positive for ${reactedToEvent.content["space.midnightthoughts.sending_user"]} on ${this.event.targetEventId ?? "unknown"}`);

        // Get the message we sent in the warning room to update it
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const warningMessage = await this.client.getEvent(this.roomId, this.event.targetEventId ?? "");
        // Update the event to reflect the action
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        warningMessage.content["space.midnightthoughts.previous_action"] = "false_positive";
        // Add relates_to to the event
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        warningMessage.content["m.relates_to"] = {
            "rel_type": "m.replace",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
            "event_id": this.event.targetEventId ?? ""
        };
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
        warningMessage.content["m.new_content"] = {
            "msgtype": "m.text",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
            "body": warningMessage.content.body,
            "format": "org.matrix.custom.html",
            // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
            "formatted_body": warningMessage.content.formatted_body
        }
        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
        await this.client.sendEvent(this.roomId, "m.room.message", warningMessage.content)
    }
}
