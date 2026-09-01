# TFM RL v2: úplne nový tréning od nuly

## Zhrnutie

- V2 začne s náhodne inicializovaným modelom, novým datasetom, novými checkpointmi a čistými metrikami.
- Nebude načítavať, porovnávať ani automaticky objavovať Gen 332 či akékoľvek staré checkpointy.
- Zachová sa použiteľná herná integrácia, legal masking, encoder a action-conditioned transformer.
- Postup bude: oprava tréningového základu → heuristický teacher → imitation learning → jeden self-play learner.
- Prvý experiment bude obsahovať iba základnú hru. Expanzie, draft, MCTS a evolúcia zostanú vypnuté.

## Implementačné zmeny

### 1. Izolované prostredie v2

- Vytvoriť samostatnú vetvu `tfm-rl-v2`.
- V2 používa vlastné adresáre pre rollouty, teacher dataset, benchmarky, checkpointy a metriky.
- Zakázať resume, bootstrap a auto-discovery starých modelov.
- Pri štarte v2 zlyhať, ak cieľový checkpoint adresár nie je prázdny, pokiaľ nebolo explicitne zvolené pokračovanie v rámci v2.
- Staré súbory sa nemusia mazať, ale žiadny v2 proces ich nebude čítať.

### 2. Oprava epizód a PPO

- Nahradiť `deque(maxlen=512)` úplnou epizódou. Bezpečnostný limit 10 000 krokov zahodí celú chybnú epizódu, nie jej začiatok.
- Rollout rozšíriť o `episode_id`, `step_index`, `policy_version`, `terminal` a `bootstrap_value`.
- Diskový store zapisuje a načítava celé epizódy; PPO batch ich nesmie preseknúť.
- GAE/returns počítať samostatne pre každú epizódu. V2 použije `gamma=1.0` a `gae_lambda=1.0`, aby konečný výsledok dostali aj skoré rozhodnutia.
- Po PPO update zvýšiť `policy_version`; zvyšné rollouty zo staršej verzie zahodiť.
- PPO reward: `{1: +1, 2: +0.25, 3: -0.25, 4: -1}` plus maximálne ±0.1 za relatívny VP rozdiel voči priemeru stola.
- Existujúci reward shaping ponechať iba ako legacy kód; vo v2 ho nastaviť na nulu.

### 3. Reprodukovateľná hra a benchmark

- Upraviť TM create-game API, aby pri `RL_ALLOW_FIXED_SEED=1` použilo seed z requestu; mimo RL režimu zostane náhodné správanie.
- Benchmarkové seedy držať v samostatnom zozname, ktorý sa nikdy nepoužije pri tréningu ani tvorbe teacher datasetu.
- Každý matchup odohrať na rovnakých seedoch so štyrmi rotáciami sedadiel.
- Reportovať completion rate, rejection count, first-place rate, priemerné poradie, relatívny VP rozdiel a 95 % interval spoľahlivosti.
- Fixné baselines budú iba `RandomLegal`, `HeuristicTeacher v1` a zmrazené champion checkpointy vytvorené v rámci v2.

### 4. Heuristický teacher

- Zaviesť rozhranie `DecisionPolicy.score_actions(state, legal_descriptors) -> PolicyDecision`.
- `PolicyDecision` obsahuje zvolenú action position/index, skóre a pravdepodobnosť každej legálnej akcie, confidence, dôvody a verziu policy.
- Implementovať `RandomLegalPolicy`, `HeuristicTeacherPolicy` a adaptér neurálnej policy.
- Teacher znovu použije existujúce heuristiky pre startup, karty a requirements.
- Doplniť scorery pre platby, map placement, štandardné projekty, míľniky, awards, konverzie zdrojov, predaj kariet a pass.
- Teacher musí vždy vybrať legálnu akciu. Nepodporovaný prompt použije deterministický fallback a zvýši metriku `teacher/fallback`.
- Počas generovania hier samplovať z teacher score distribution s nízkou teplotou, aby dataset obsahoval viac než jednu deterministickú trajektóriu.

### 5. Teacher dataset a tvoje opravy

- Zaviesť shardovaný formát `teacher_sample.v1`: planner bundle, legálne action descriptory, teacher probabilities, confidence, seed, epizóda/krok a konečný výsledok.
- Dataset deliť podľa celých hier na 80 % train, 10 % validation a 10 % test.
- Decision Explainer rozšíriť o označenie jednej alebo viacerých prijateľných akcií, poznámku a možnosť `skip`.
- Na ručnú kontrolu prioritizovať pozície s nízkou teacher confidence a najväčším rozdielom medzi teacherom a študentom.
- Human label má tréningovú váhu 4.0, high-confidence teacher 1.0 a low-confidence teacher 0.25.
- Pri viacerých prijateľných akciách rozdeliť cieľovú pravdepodobnosť rovnomerne.
- Pred prvým pretrainingom nazbierať minimálne 100 000 teacher rozhodnutí a 100 ručne skontrolovaných strategických pozícií.

### 6. Supervised pretraining

- Model inicializovať úplne náhodne; nepoužiť žiadne staré weights ani optimizer state.
- Policy trénovať pomocou KL/cross-entropy nad variabilným legal-action maskom.
- Value head trénovať na konečný rank reward a relatívny VP výsledok teacher hier.
- Ukladať samostatné BC checkpointy a validačné metriky.
- PPO sa nesmie spustiť, kým checkpoint nedosiahne aspoň 85 % top-1 a 97 % top-3 accuracy na teacher test sete.
- Na minimálne 100 human-labeled pozíciách požadovať aspoň 80 % top-3 accuracy.

### 7. Curriculum a self-play

- Stage 0: štyria hráči, Tharsis, beginner corporation, bez draftu, bez expanzií a bez map shuffle.
- Stage 1: Tharsis, Corporate Era, normálna voľba corporation z dvoch ponúk, stále bez draftu a expanzií.
- Trénovať iba jedného main learnera. Súperi sú zmrazení a nezbierajú PPO rollouty.
- Pred Stage 0 gate samplovať súperov 50 % Teacher a 50 % RandomLegal.
- Po Stage 0 gate používať 40 % Teacher, 40 % aktuálny frozen v2 champion a 20 % staršie v2 champion checkpointy.
- Kandidáta vyhodnotiť každých 25 000 nových rozhodnutí.
- Nový champion povýšiť iba pri splnení completion/rejection kontrol a bez regresie voči aktuálnemu championovi.
- Evolúciu neurónových weights ponechať vypnutú.

## Testovanie a akceptácia

- Unit testy overia oddelené GAE epizódy, neprerušované diskové shardy, terminal return pre skoré kroky a filtrovanie starých policy verzií.
- Teacher testy pokryjú všetky action families, deterministické skóre a výber výhradne z legal masku.
- Dataset testy overia schému, human weights, atomické shardy a neprítomnosť seed/game leakage.
- Server test potvrdí identický počiatočný stav pri rovnakom seede a odlišný stav pri inom seede.
- Smoke test: 100 hier, completion rate minimálne 99 %, nulové serverové rejectiony a teacher fallback pod 5 %.
- Stage 0 gate: 30 seedov × 4 sedadlá proti trom RandomLegal súperom; first-place rate minimálne 55 %.
- Stage 1 finálny gate: 30 seedov × 4 sedadlá proti trom Teacher v1 súperom; spodná hranica 95 % Wilson intervalu first-place rate musí byť vyššia než náhodných 25 %.
- Regression gate: kandidát musí proti aktuálnemu v2 championovi dosiahnuť pairwise score aspoň 50 % bez poklesu completion rate.

## Predpoklady

- „Od nuly“ znamená nové modelové weights, optimizer, dataset, rollouty, champion pool, rating aj benchmark históriu; existujúca herná infraštruktúra sa znovu použije.
- Staré checkpointy ostanú fyzicky nedotknuté, ale v2 o nich nebude vedieť.
- Tréning a testy pobežia v Docker prostredí s PyTorch.
- Prelude, draft a ďalšie expanzie sa pridajú až po splnení Stage 1 gate.
- MCTS alebo AlphaZero-like search sa začne riešiť až po preukázateľnom prekonaní heuristického teachera.
