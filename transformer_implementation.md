Implementácia Attention mechanizmu pre synergie kariet
Tento plán popisuje prechod z aktuálnej heuristickej agregácie tagov na dynamický Attention mechanizmus (Transformer-like architektúra).

Navrhované Zmeny
1. Card Embedding Vrstva
Namiesto manuálneho počítania tagov vytvoríme CardEmbedding modul, ktorý priradí každej karte (meno + tagy + cena) hustý vektor (napr. 64 dimenzií).

2. Tableau Encoder (Self-Attention)
Využijeme Multi-Head Self-Attention na spracovanie vyložených kariet (Tableau).

Query/Key/Value: Atribúty každej vyloženej karty.
Výsledok: Jedna karta v tableau bude "vedieť" o všetkých ostatných silných synergiách (napr. Science karta vie o prítomnosti Olympus Conference).
3. Hand-Tableau Interaction (Cross-Attention)
Karty na ruke (Hand) alebo v drafte budú vystupovať ako Queries a Tableau ako Keys/Values.

Model priamo vyhodnotí, na ktoré vyložené karty "upriamuje pozornosť" daná karta na ruke.
4. Draft Fáza a "Hate-drafting"
V drafte Attention mechanizmus porovnáva ponúkané karty nielen s tvojím Tableau, ale aj s Tableau súperov.

Ak karta dáva VP za Jovian tagy a ty ich máš 3, Cross-Attention medzi touto kartou a tvojím Tableau vygeneruje silný signál ("High Synergy").
Zároveň, ak má súper 5 Jovian tagov, Attention mechanizmus na súperovo Tableau ukáže, že táto karta má pre neho extrémnu hodnotu. Model sa tak naučí "Hate-drafting" – zobrať kartu len preto, aby ju nemal súper.
Technická Implementácia (Ukážka)
[MODIFY] 
agent.py
Pridanie Transformer vrstiev do 
TerraformingMarsNetwork
:

class CardAttentionModule(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
    def forward(self, tableau_embeds):
        # tableau_embeds shape: [Batch, NumCards, EmbedDim]
        # Transformer vyžaduje [SeqLen, Batch, EmbedDim]
        x = tableau_embeds.transpose(0, 1)
        x = self.transformer(x)
        return x.transpose(0, 1)
[MODIFY] 
state_encoder.py
Zmena kódovania stavu:

Namiesto List[float] (vektor 1024) bude 
encode()
 vracať tensor so sekvenciou kariet.
Dôležité: Zachováme spätú kompatibilitu pre globálne parametre (kyslík, teplo), ktoré ostanú v MLP časti.
Overovací Plán
Automatizované Testy
pytest tests/test_attention_shape.py: Overenie správnosti rozmerov (tensors).
Sledovanie synergy_score v 
agent_1_config.json
 (očakávame nárast po natrénovaní).