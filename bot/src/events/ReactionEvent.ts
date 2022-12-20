import { RoomEvent } from "matrix-bot-sdk";

/**
 * The defintion for the relation
 * @category Matrix event contents
 * @see ReactionEventContent
 */
export interface RelatesTo {
    rel_type: "m.annotation";
    event_id: string;
    key: string;
}

/**
 * The content definition for m.reaction events
 * @category Matrix event contents
 * @see ReactionEvent
 */
export interface ReactionEventContent {
    "m.relates_to": RelatesTo | undefined;
}

/**
 * Represents an m.reaction room event
 * @category Matrix events
 */
export class ReactionEvent<T extends ReactionEventContent> extends RoomEvent<T> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    constructor(event: any) {
        super(event);
    }

    /**
     * Whether or not the event is redacted (or looked redacted).
     */
    public get isRedacted(): boolean {
        // Presume the event redacted if we're missing content
        return this.content["m.relates_to"] === undefined;
    }

    public get relatesTo(): RelatesTo {
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion, @typescript-eslint/no-extra-non-null-assertion
        return this.content["m.relates_to"]!!;
    }
}
