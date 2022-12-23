import { MatrixClient, MatrixGlob, RoomCreateOptions } from "matrix-bot-sdk";
import { Config } from "../index";
import short from "short-uuid";

export enum EntityType {
    /// `entity` is to be parsed as a glob of users IDs
    RULE_USER = "m.policy.rule.user",

    /// `entity` is to be parsed as a glob of room IDs/aliases
    RULE_ROOM = "m.policy.rule.room",

    /// `entity` is to be parsed as a glob of server names
    RULE_SERVER = "m.policy.rule.server",
}

export const RULE_USER = EntityType.RULE_USER;
export const RULE_ROOM = EntityType.RULE_ROOM;
export const RULE_SERVER = EntityType.RULE_SERVER;

// README! The order here matters for determining whether a type is obsolete, most recent should be first.
// These are the current and historical types for each type of rule which were used while MSC2313 was being developed
// and were left as an artifact for some time afterwards.
// Most rules (as of writing) will have the prefix `m.room.rule.*` as this has been in use for roughly 2 years.
export const USER_RULE_TYPES = [RULE_USER, "m.room.rule.user", "org.matrix.mjolnir.rule.user"];
export const ROOM_RULE_TYPES = [RULE_ROOM, "m.room.rule.room", "org.matrix.mjolnir.rule.room"];
export const SERVER_RULE_TYPES = [RULE_SERVER, "m.room.rule.server", "org.matrix.mjolnir.rule.server"];
export const ALL_RULE_TYPES = [...USER_RULE_TYPES, ...ROOM_RULE_TYPES, ...SERVER_RULE_TYPES];

export enum Recommendation {
    /// The rule recommends a "ban".
    ///
    /// The actual semantics for this "ban" may vary, e.g. room ban,
    /// server ban, ignore user, etc. To determine the semantics for
    /// this "ban", clients need to take into account the context for
    /// the list, e.g. how the rule was imported.
    Ban = "m.ban",

    /// The rule specifies an "opinion", as a number in [-100, +100],
    /// where -100 represents a user who is considered absolutely toxic
    /// by whoever issued this ListRule and +100 represents a user who
    /// is considered absolutely absolutely perfect by whoever issued
    /// this ListRule.
    Opinion = "org.matrix.msc3845.opinion",

    /**
     * This is a rule that recommends allowing a user to participate.
     * Used for the construction of allow lists.
     */
    Allow = "org.matrix.mjolnir.allow",
}

interface PolicyStateEvent {
    type: string,
    content: {
        entity: string,
        reason: string,
        recommendation: Recommendation,
        opinion?: number,
    },
    event_id: string,
    state_key: string,
}

/**
 * All variants of recommendation `m.ban`
 */
const RECOMMENDATION_BAN_VARIANTS = [
    // Stable
    Recommendation.Ban,
    // Unstable prefix, for compatibility.
    "org.matrix.mjolnir.ban"
];

/**
 * All variants of recommendation `m.opinion`
 */
const RECOMMENDATION_OPINION_VARIANTS: string[] = [
    // Unstable
    Recommendation.Opinion
];

const RECOMMENDATION_ALLOW_VARIANTS: string[] = [
    // Unstable
    Recommendation.Allow
]

export const OPINION_MIN = -100;
export const OPINION_MAX = +100;

/**
 * Representation of a rule within a Policy List.
 */
export abstract class ListRule {
    /**
     * A glob for `entity`.
     */
    private glob: MatrixGlob;
    constructor(
        /**
         * The event source for the rule.
         */
        public readonly sourceEvent: PolicyStateEvent,
        /**
         * The entity covered by this rule, e.g. a glob user ID, a room ID, a server domain.
         */
        public readonly entity: string,
        /**
         * A human-readable reason for this rule, for audit purposes.
         */
        public readonly reason: string,
        /**
         * The type of entity for this rule, e.g. user, server domain, etc.
         */
        public readonly kind: EntityType,
        /**
         * The recommendation for this rule, e.g. "ban" or "opinion", or `null`
         * if the recommendation is one that Mjölnir doesn't understand.
         */
        public readonly recommendation: Recommendation | null) {
        this.glob = new MatrixGlob(entity);
    }

    /**
     * Determine whether this rule should apply to a given entity.
     */
    public isMatch(entity: string): boolean {
        return this.glob.test(entity);
    }

    /**
     * @returns Whether the entity in he rule represents a Matrix glob (and not a literal).
     */
    public isGlob(): boolean {
        return /[*?]/.test(this.entity);
    }

    /**
     * Validate and parse an event into a ListRule.
     *
     * @param event An *untrusted* event.
     * @returns null if the ListRule is invalid or not recognized by Mjölnir.
     */
    public static parse(event: PolicyStateEvent): ListRule | null {
        // Parse common fields.
        // If a field is ill-formed, discard the rule.
        const content = event['content'];
        if (!content || typeof content !== "object") {
            return null;
        }
        const entity = content['entity'];
        if (!entity || typeof entity !== "string") {
            return null;
        }
        const recommendation = content['recommendation'];
        if (!recommendation || typeof recommendation !== "string") {
            return null;
        }

        const reason = content['reason'] || '<no reason>';
        if (typeof reason !== "string") {
            return null;
        }

        const type = event['type'];
        let kind;
        if (USER_RULE_TYPES.includes(type)) {
            kind = EntityType.RULE_USER;
        } else if (ROOM_RULE_TYPES.includes(type)) {
            kind = EntityType.RULE_ROOM;
        } else if (SERVER_RULE_TYPES.includes(type)) {
            kind = EntityType.RULE_SERVER;
        } else {
            return null;
        }

        // From this point, we may need specific fields.
        if (RECOMMENDATION_BAN_VARIANTS.includes(recommendation)) {
            return new ListRuleBan(event, entity, reason, kind);
        } else if (RECOMMENDATION_OPINION_VARIANTS.includes(recommendation)) {
            const opinion = content['opinion'];
            if (!Number.isInteger(opinion)) {
                return null;
            }
            return new ListRuleOpinion(event, entity, reason, kind, opinion);
        } else if (RECOMMENDATION_ALLOW_VARIANTS.includes(recommendation)) {
            return new ListRuleAllow(event, entity, reason, kind);
        } else {
            // As long as the `recommendation` is defined, we assume
            // that the rule is correct, just unknown.
            return new ListRuleUnknown(event, entity, reason, kind, content);
        }
    }
}

/**
 * A rule representing a "ban".
 */
export class ListRuleBan extends ListRule {
    constructor(
        /**
         * The event source for the rule.
         */
        sourceEvent: PolicyStateEvent,
        /**
         * The entity covered by this rule, e.g. a glob user ID, a room ID, a server domain.
         */
        entity: string,
        /**
         * A human-readable reason for this rule, for audit purposes.
         */
        reason: string,
        /**
         * The type of entity for this rule, e.g. user, server domain, etc.
         */
        kind: EntityType,
    ) {
        super(sourceEvent, entity, reason, kind, Recommendation.Ban)
    }
}

/**
 * A rule representing an "allow".
 */
export class ListRuleAllow extends ListRule {
    constructor(
        /**
         * The event source for the rule.
         */
        sourceEvent: PolicyStateEvent,
        /**
         * The entity covered by this rule, e.g. a glob user ID, a room ID, a server domain.
         */
        entity: string,
        /**
         * A human-readable reason for this rule, for audit purposes.
         */
        reason: string,
        /**
         * The type of entity for this rule, e.g. user, server domain, etc.
         */
        kind: EntityType,
    ) {
        super(sourceEvent, entity, reason, kind, Recommendation.Allow)
    }
}

/**
 * A rule representing an "opinion"
 */
export class ListRuleOpinion extends ListRule {
    constructor(
        /**
         * The event source for the rule.
         */
        sourceEvent: PolicyStateEvent,
        /**
         * The entity covered by this rule, e.g. a glob user ID, a room ID, a server domain.
         */
        entity: string,
        /**
         * A human-readable reason for this rule, for audit purposes.
         */
        reason: string,
        /**
         * The type of entity for this rule, e.g. user, server domain, etc.
         */
        kind: EntityType,
        /**
         * A number in [-100, +100] where -100 represents the worst possible opinion
         * on the entity (e.g. toxic user or community) and +100 represents the best
         * possible opinion on the entity (e.g. pillar of the community).
         */
        public readonly opinion: number | undefined
    ) {
        super(sourceEvent, entity, reason, kind, Recommendation.Opinion);
        if (!Number.isInteger(opinion)) {
            throw new TypeError(`The opinion must be an integer, got ${opinion ?? 'undefined'}`);
        }
        if ((opinion ?? 0) < OPINION_MIN || (opinion ?? 0) > OPINION_MAX) {
            throw new TypeError(`The opinion must be within [-100, +100], got ${opinion ?? 'undefined'}`);
        }
    }
}

/**
 * Any list rule that we do not understand.
 */
export class ListRuleUnknown extends ListRule {
    constructor(
        /**
         * The event source for the rule.
         */
        sourceEvent: PolicyStateEvent,
        /**
         * The entity covered by this rule, e.g. a glob user ID, a room ID, a server domain.
         */
        entity: string,
        /**
         * A human-readable reason for this rule, for audit purposes.
         */
        reason: string,
        /**
         * The type of entity for this rule, e.g. user, server domain, etc.
         */
        kind: EntityType,
        /**
         * The event used to create the rule.
         */
        public readonly content: unknown,
    ) {
        super(sourceEvent, entity, reason, kind, null);
    }
}

/* Soom of this code is taken from. (Not copied but based upon. This is needed for compat reasons.)  */
export default class BanListHandler {
    private readonly uuidGen = short(short.constants.cookieBase90);
    constructor(private readonly client: MatrixClient, private readonly config: Config) { }

    /**
     * This is used to annotate state events we store with the rule they are associated with.
     * If we refactor this, it is important to also refactor any listeners to 'PolicyList.update'
     * which may assume `ListRule`s that are removed will be identital (Object.is) to when they were added.
     * If you are adding new listeners, you should check the source event_id of the rule.
     */
    private static readonly EVENT_RULE_ANNOTATION_KEY = 'org.matrix.mjolnir.annotation.rule';

    public static async createPolicyRoom(client: MatrixClient): Promise<string> {
        const powerLevels: { [key: string]: number | object } = {
            "ban": 50,
            "events": {
                "m.room.name": 100,
                "m.room.power_levels": 100,
            },
            "events_default": 50, // non-default
            "invite": 0,
            "kick": 50,
            "notifications": {
                "room": 20,
            },
            "redact": 50,
            "state_default": 50,
            "users": {
                [await client.getUserId()]: 100,
            },
            "users_default": 0,
        };
        // Support for MSC3784.
        const roomOptions: RoomCreateOptions = {
            creation_content: {
                type: "support.feline.policy.lists.msc.v1"
            },
            preset: "public_chat",

            power_level_content_override: powerLevels,
        }
        const listRoomId = await client.createRoom(roomOptions);
        return listRoomId
    }

    // Adds a rule to the ban list and bans the user.
    public async banUser(userId: string, reason: string): Promise<void> {
        await this.addRule(userId, reason, RULE_USER, Recommendation.Ban);

        // Get all rooms the user is in
        const rooms = await this.client.getJoinedRooms();
        for (const room of rooms) {
            // Get all members in the room
            const members = await this.client.getJoinedRoomMembers(room);
            for (const member of members) {
                // If the member is the user we want to ban
                if (member == userId) {
                    // Ban the user
                    // TODO: allow setting a reason
                    await this.client.banUser(room, member, reason);

                    // Redact all messages the user sent
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
                    const messages = await this.client.doRequest("GET", `/_matrix/client/v3/rooms/${room}/messages?dir=b&limit=100`);
                    // Load more if we have more than 100 messages
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                    if (messages.chunk.length == 100) {
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/restrict-template-expressions
                        const moreMessages = await this.client.doRequest("GET", `/_matrix/client/v3/rooms/${room}/messages?dir=b&limit=100&from=${messages.end}`);
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
                        messages.chunk = messages.chunk.concat(moreMessages.chunk);
                    }
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                    for (const message of messages.chunk) {
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access
                        await this.client.redactEvent(room, message.event_id);
                    }
                }
            }
        }
    }

    // This kicks the user. It doesn't actually add a rule to the ban list.
    public async kickUser(userId: string, reason: string): Promise<void> {
        // Get all rooms the user is in
        const rooms = await this.client.getJoinedRooms();
        for (const room of rooms) {
            // Get all members in the room
            const members = await this.client.getJoinedRoomMembers(room);
            for (const member of members) {
                // If the member is the user we want to ban
                if (member == userId) {
                    // Ban the user
                    // TODO: allow setting a reason
                    await this.client.kickUser(room, member, reason);

                    // Redact all messages the user sent
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
                    const messages = await this.client.doRequest("GET", `/_matrix/client/v3/rooms/${room}/messages?dir=b&limit=100`);
                    // Load more if we have more than 100 messages
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                    if (messages.chunk.length == 100) {
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/restrict-template-expressions
                        const moreMessages = await this.client.doRequest("GET", `/_matrix/client/v3/rooms/${room}/messages?dir=b&limit=100&from=${messages.end}`);
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call
                        messages.chunk = messages.chunk.concat(moreMessages.chunk);
                    }
                    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                    for (const message of messages.chunk) {
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-member-access
                        await this.client.redactEvent(room, message.event_id);
                    }
                }
            }
        }
    }

    private async addRule(userId: string, reason: string, kind: string, recommendation: Recommendation): Promise<void> {
        await this.client.sendStateEvent(this.config.banlistRoom, kind, this.uuidGen.new(), {
            entity: userId,
            reason: reason,
            recommendation: recommendation,
        });
    }

    /**
     * Return all the active rules of a given kind.
     * @param kind e.g. RULE_SERVER (m.policy.rule.server). Rule types are always normalised when they are interned into the PolicyList.
     * @param recommendation A specific recommendation to filter for e.g. `m.ban`. Please remember recommendation varients are normalized.
     * @returns The active ListRules for the ban list of that kind.
     */
    public async rulesOfKind(roomId: string, kind: string, recommendation?: Recommendation): Promise<ListRule[]> {
        const rules: ListRule[] = []
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
        const stateKeyMap = await this.client.getRoomStateEvent(roomId, kind, "");
        if (stateKeyMap) {
            // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-call
            for (const event of stateKeyMap.values()) {
                // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment
                const rule = event[BanListHandler.EVENT_RULE_ANNOTATION_KEY];
                // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                if (rule && rule.kind === kind) {
                    if (recommendation === undefined) {
                        rules.push(rule as ListRuleUnknown);
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                    } else if (rule.recommendation === recommendation) {
                        // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
                        switch (rule.recommendation) {
                            case Recommendation.Ban:
                                rules.push(rule as ListRuleBan);
                                break;
                            case Recommendation.Allow:
                                rules.push(rule as ListRuleAllow);
                                break;
                            case Recommendation.Opinion:
                                rules.push(rule as ListRuleOpinion);
                                break;
                            default:
                                rules.push(rule as ListRuleUnknown);
                                break;
                        }
                    }
                }
            }
        }
        return rules;
    }

    public async serverRules(roomId: string): Promise<ListRule[]> {
        return this.rulesOfKind(roomId, RULE_SERVER);
    }

    public async userRules(roomId: string): Promise<ListRule[]> {
        return this.rulesOfKind(roomId, RULE_USER);
    }

    public async roomRules(roomId: string): Promise<ListRule[]> {
        return this.rulesOfKind(roomId, RULE_ROOM);
    }

    public async allRules(roomId: string): Promise<ListRule[]> {
        return [...await this.serverRules(roomId), ...await this.userRules(roomId), ...await this.roomRules(roomId)];
    }
}
