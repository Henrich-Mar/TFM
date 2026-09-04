Paste this into the RL repository as the upgrade checklist:
# Terraforming Mars upstream upgrade: RL integration delta

Base upstream: `origin/main` at `d0d9c43` (2026-09-03)  
RL merge commit: `55bdd2dda5`

## Required server changes

### 1. Deterministic game creation for RL

Patch `src/server/routes/ApiCreateGame.ts`.

- Import `SeededRandom` from `src/common/utils/Random`.
- Add `ApiCreateGame.resolveGameSeed(requestedSeed)`.
- Enable fixed seeds only when `RL_ALLOW_FIXED_SEED` is one of:
  `1`, `true`, `yes`, `on`.
- If disabled, ignore the submitted seed and use `Math.random()`.
- Accept fractional seeds in `[0, 1)`.
- Convert integer seeds to the low 32-bit fraction:

```ts
const uint32Seed = Math.trunc(parsed) >>> 0;
return uint32Seed / 0x100000000;
Use the resolved seed both for random-board selection and Game.newInstance.
Current-upstream-compatible placement:
const seed = ApiCreateGame.resolveGameSeed(gameReq.seed);
const boards = ApiCreateGame.boardOptions(gameReq.board);
gameReq.board = boards[new SeededRandom(seed).nextInt(boards.length)];

// Current upstream argument order:
game = Game.newInstance(
  gameId,
  players,
  players[firstPlayerIdx],
  spectatorId,
  gameOptions,
  seed,
);
Add/retain a test proving equal requested seeds create equal deck signatures when RL_ALLOW_FIXED_SEED=1.
2. Human demonstration capture
Add src/server/training/HumanMoveListener.ts.
Behavior:
Activated only when TFM_HUMAN_LISTENER_URL is non-empty.
Optional filter: TFM_HUMAN_LISTENER_PLAYER_IDS, comma-separated player IDs.Empty/unset means capture all players.

Optional auth header:
X-TFM-Human-Listener-Token: $TFM_HUMAN_LISTENER_TOKEN
POST delivery is fire-and-forget; collector failure must never reject or delay a game action.
Abort delivery after 750 ms.
Log delivery failure at most once per 60 seconds.
Capture payload before processing the action:
{
  "schema_version": "tfm.human_move.v1",
  "captured_at": "ISO-8601 timestamp",
  "game_id": "…",
  "player_id": "…",
  "player_name": "…",
  "player_state": "<Server.getPlayerModel(player)>",
  "response": "<accepted InputResponse>"
}
After the game ends, emit:
{
  "schema_version": "tfm.human_move.v1",
  "event_type": "game_complete",
  "game_id": "…",
  "player_state": "<Server.getPlayerModel(player)>"
}
Patch src/server/routes/PlayerInput.ts at the non-undo branch:
const humanMoveCapture = prepareHumanMoveCapture(player, entity);
player.process(entity);
deliverHumanMove(humanMoveCapture);
captureHumanGameCompletion(player);
responses.writeJson(res, ctx, Server.getPlayerModel(player));
Important: prepare the snapshot before player.process, but deliver it only after player.process succeeds. Do not capture undo requests.
3. RL server recycle endpoint
Add src/server/routes/ApiRlRecycle.ts.
Route: /api/rl/recycle
Add API_RL_RECYCLE: 'api/rl/recycle' to src/common/app/paths.ts.
Register it in src/server/server/requestProcessor.ts.
Authorization:
Endpoint is hidden as 404 unless:RL_RECYCLE_ENABLED=1
RL_CONTROL_TOKEN is configured

Caller sends X-RL-Control-Token.
Compare tokens with crypto.timingSafeEqual.
GET returns {"status":"ready"}.
Authorized POST returns HTTP 202 with {"status":"recycling"}, then exits the process after ~50 ms.
Intended only for an RL-specific Docker service with restart policy and disposable/tmpfs SQLite storage.
4. SQLite participant de-duplication
Patch SQLite.storeParticipants:
De-duplicate entry.participantIds with Array.from(new Set(...)).
Return immediately when no participants exist.
Preserve ON CONFLICT (game_id, participant) DO NOTHING.
This makes repeated saves/recycles idempotent.
RL client/trainer contract changes from upstream
Payments are now megacredits, not megaCredits
Update both action generation and observation parsing:
payment.megaCredits → payment.megacredits
player.megaCredits → player.megacredits
player.megaCreditProduction → player.megacreditProduction
This applies to both action forms:
{ "type": "payment", "payment": { "…": 0 } }
{ "type": "projectCard", "card": "…", "payment": { "…": 0 } }
Always emit a complete payment object
Current server validation requires every spendable-resource key to be present:
const emptyPayment = {
  megacredits: 0,
  heat: 0,
  steel: 0,
  titanium: 0,
  plants: 0,
  microbes: 0,
  floaters: 0,
  lunaArchivesScience: 0,
  spireScience: 0,
  seeds: 0,
  auroraiData: 0,
  graphene: 0,
  kuiperAsteroids: 0,
};
The trainer should clone this object and set only the resources it wants to spend. Do not send legacy megaCredits.
Input/observation surface expanded
Payment-related input models may now expose:
floaters
microbes
graphene
auroraiData
spireScience
reserveUnits
The RL action decoder should treat these as possible payment resources where allowed by paymentOptions.
Files to copy/adapt
Required:
src/server/routes/ApiCreateGame.ts
src/server/routes/PlayerInput.ts
src/server/training/HumanMoveListener.ts (new)
src/server/routes/ApiRlRecycle.ts (new, if recycle is used)
src/common/app/paths.ts
src/server/server/requestProcessor.ts
Recommended:
src/server/database/SQLite.ts
tests/routes/ApiCreateGame.spec.ts
tests/database/SQLite.spec.ts
.dockerignore additions: .git, build, node_modules
Validation
Fixed seed: same RL seed yields the same board/deck; different seed changes it.
With RL_ALLOW_FIXED_SEED unset, supplied seeds do not make public games deterministic.
Human listener receives pre-action private player state and accepted action.
Listener outage does not affect player input success.
Payment and project-card actions use megacredits and include every payment key.
Recycle endpoint is 404 when disabled, 401 on bad token, and 202 on an authorized POST.