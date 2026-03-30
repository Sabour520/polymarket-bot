from py_clob_client.client import ClobClient
from py_clob_client.clob_types import TradeParams
from dotenv import load_dotenv
import os

load_dotenv()

client = ClobClient(
    os.getenv('POLYMARKET_HOST', 'https://clob.polymarket.com'),
    key=os.getenv('POLYMARKET_PRIVATE_KEY'),
    chain_id=int(os.getenv('POLYMARKET_CHAIN_ID', '137'))
)
client.set_api_creds(client.create_or_derive_api_creds())

tokens_a_tester = [
    ("West Ham YES", "11053886078271921641197817628012876432166512653115577682436598914104932954889"),
    ("TISZA YES",    "61963573353880265694358880181512412652951688509792017552668784297365342759317"),
    ("Trae Young YES", "29593985704462432449808466045226728204370351622211005452559129942249477981400"),
]

for nom, token_id in tokens_a_tester:
    try:
        params = TradeParams(maker_asset_id=token_id)
        trades = client.get_trades(params)
        nb = len(trades) if trades else 0
        print(f"{nom} : {nb} trades")
        if nb > 0:
            print(f"  Exemple : {trades[0]}")
            print(f"  TOKEN_ID A UTILISER : {token_id}")
            break
    except Exception as e:
        print(f"{nom} : erreur - {e}")
