# Terraforming Mars RL: Hĺbková Technická Architektúra

Tento dokument poskytuje podrobný pohľad na fungovanie Reinforcement Learning (RL) prostredia pre hru Terraforming Mars.

---

## 1. Tréningový Cyklus (Generácie a Turnaje)
Tréning prebieha v cykloch nazývaných **Generácie**. Každá generácia pozostáva z:
1.  **Turnajov:** Agenti z aktuálnej populácie hrajú hry proti sebe (self-play).
2.  **Zberu dát (Rollouts):** Počas hier si každý agent ukladá prechody (stav -> akcia -> odmena) do svojho bufferu.
3.  **Optimalizácie (PPO Update):** Po skončení turnajov sa neurónové siete natrénujú na zozbieraných dátach.
4.  **Evolúcie a Gating:** Agenti, ktorí nespĺňajú určité kvalitatívne kritériá (napr. minimálny počet zahraných kariet), sú penalizovaní alebo vyradení.

---

## 2. PPO (Proximal Policy Optimization)
Algoritmus PPO zabezpečuje stabilné učenie pomocou:
*   **Clipping (Orezávanie):** Zabraňuje príliš veľkým zmenám v stratégii (policy), ktoré by mohli rozbiť model.
*   **GAE (Generalized Advantage Estimation):** Odhaduje, o koľko lepšia bola zvolená akcia oproti priemeru v danom stave.
*   **Rare State Priority:** Špeciálny mechanizmus, ktorý dáva vyššiu váhu vzácnym situáciám (napr. financovanie ocenení/awardov alebo dôležité míľniky), aby sa ich model naučil riešiť efektívnejšie.

---

## 3. Scoring a Reward Shaping (Odmeny)
Systém nepoužíva len koncové víťazstvo/prehru, ale aj "husté" odmeny (dense rewards) počas hry:
*   **Terminal Reward:** Hlavná odmena na konci hry založená na poradí (Rank) a Victory Points.
*   **Step Reward Decomposition:** Každý ťah je vyhodnotený podľa:
    *   **TR Delta:** Zvýšenie Terraform Ratingu.
    *   **VP Units:** Získanie budúcich bodov z kariet, míľnikov a ocenení.
    *   **Board Tactics:** Bonusy za mestá obklopené zeleňou a penalizácie za umiestnenie zelene vedľa súperovho mesta.
    *   **Resource Utilization:** Odmena za efektívne míňanie ocele a titánu (stláčanie "tlaku" zdrojov).
    *   **Passivity Penalty:** Malá penalizácia za pasivitu alebo neefektívne predávanie patentov.

---

## 4. Vector Awards & Aux Heads (Pomocné hlavy)
Model má okrem hlavného rozhodovania aj tzv. **Auxiliary Heads**. Sú to "pomocné senzory", ktoré predpovedajú dôležité faktory v hre:
*   **Milestone Claimability (Vector 70):** Model predpovedá šancu na získanie každého zo 70 možných míľnikov.
*   **Award EV:** Odhad očakávanej hodnoty bodov z ocenení (Awards).
*   **Playable Cards:** Odhad, koľko kariet v ruke je momentálne hrateľných.
*   **Resource Targets:** Predpovedá ciele pre oceľ a titán.

Tieto predpovede nútia neurónovú sieť lepšie "pochopiť" hlbokú štruktúru hry, aj keď tieto čísla priamo nepoužíva na vykonanie ťahu. Funguje to ako doplnkový tréning pre intuíciu.

---

## 5. League System
Aby sa agenti nezasekli v uzavretej bubline (overfitting), systém používa ligové bazény:
*   **Main Pool:** Aktuálne najlepší agenti.
*   **Historical Pool:** Staré verzie agentov (prevencia "zabúdania" stratégií).
*   **Exploiter Pool:** Agenti špecializovaní na hľadanie slabín v aktuálnej stratégii.

---

## 6. Model Architecture (GRU & Phased Heads)
*   **Recurrent Memory (GRU):** Model vníma históriu hry, nielen aktuálny moment.
*   **Phase-Specific Heads:** Rôzne časti mozgu sa aktivujú pri rôznych fázach (Draft, Akcia, Produkcia), čo umožňuje jemnejšiu špecializáciu.
*   **State Encoder:** Prekladá stovky parametrov hry (mapa, zdroje, súperove tableau) do kompaktnej formy pre mozog.

---

## 7. Budúce Možné Vylepšenia
Na základe analýzy sme identifikovali niekoľko oblastí pre ďalší rozvoj:
*   **Adaptívne PPO Hyperparametre:** Implementácia **Adaptive KL Penalty** by automaticky regulovala rýchlosť zmien v stratégii, čím by sa zvýšila stabilita tréningu.
*   **Dynamické Reward Annealing:** Postupné vypínanie pomocných odmien (napr. za TR alebo zdroje) v neskorších fázach tréningu, aby sa agent sústredil čisto na finálne Victory Points.
*   **Attention Mechanizmus:** Využitie Transformer-like architektúry pre spracovanie vyložených kariet (Tableau), čo by umožnilo lepšie zachytiť zložité synergie medzi kartami.
*   **Modelovanie Súperov:** Pridanie pomocnej hlavy na predpovedanie akcií a produkcie súperov pre lepšiu reaktívnu (counter-play) stratégiu.
*   **Dynamické Gating Percentily:** Nahradenie fixných limitov pre postup agentov dynamickými percentilmi (napr. top 70% populácie), čo by zabezpečilo plynulejšiu evolúciu.
*   **Aktívny Benchmarking:** Pravidelné testovanie proti "Boss" agentom (historicky najlepšie verzie alebo heuristické boty) pre objektívne meranie progresu.
