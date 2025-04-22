import bz2
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from betfairlightweight import StreamListener
from betfairlightweight.streaming.stream import HistoricalStream

# === Settings ===
bz2_file_path = "path/to/your/file.bz2"
snapshot_interval = timedelta(hours=1)

# === Decompress the file ===
with bz2.open(bz2_file_path, 'rb') as bz2_file:
    temp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_json.write(bz2_file.read())
    temp_json_path = temp_json.name

# === Set up the stream ===
listener = StreamListener()
stream = HistoricalStream(file_path=temp_json_path, listener=listener)
gen = stream.get_generator()

# === Initialize ===
last_snapshot_time = {}

rows = []

for market_books in gen():
    for market_book in market_books:
        market_id = market_book.market_id
        pub_time = datetime.fromtimestamp(market_book.publish_time_epoch / 1000)

        # Only snapshot if 1 hour has passed since last capture for this market
        if market_id not in last_snapshot_time or pub_time - last_snapshot_time[market_id] >= snapshot_interval:
            last_snapshot_time[market_id] = pub_time

            for runner in market_book.runners:
                back_prices = runner.ex.available_to_back
                lay_prices = runner.ex.available_to_lay

                rows.append({
                    "market_id": market_id,
                    "timestamp": pub_time,
                    "selection_id": runner.selection_id,
                    "back_1_price": back_prices[0].price if back_prices else None,
                    "lay_1_price": lay_prices[0].price if lay_prices else None,
                    "ltp": runner.last_price_traded,
                    "total_volume": runner.total_matched
                })

# === Convert to DataFrame ===
df = pd.DataFrame(rows)

# Optional: sort and inspect
df.sort_values(["market_id", "timestamp"], inplace=True)
print(df.head())
