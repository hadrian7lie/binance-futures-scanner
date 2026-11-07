
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

BASE = "https://fapi.binance.com"
TOP_COINS = 30
MIN_SCORE = 65
TIMEOUT = 12

session = requests.Session()
session.headers.update({"User-Agent": "Binance-Futures-Scanner-V7"})


def api(path, params=None):
    for attempt in range(3):
        try:
            r = session.get(
                BASE + path, params=params, timeout=TIMEOUT
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == 2:
                print("API error:", path, str(e)[:150])
                return None
            time.sleep(2)


def candles(symbol, interval, limit=200):
    data = api("/fapi/v1/klines", {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    })
    if not data:
        return None

    df = pd.DataFrame(data, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy", "taker_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna()


def indicators(df):
    df = df.copy()
    close = df["close"]

    df["ema20"] = close.ewm(span=20, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["ema200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()
    avg_loss = loss.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    df["vol_avg"] = df["volume"].rolling(20).mean()
    df["vol_x"] = (
        df["volume"] / df["vol_avg"].replace(0, float("nan"))
    ).fillna(0)

    return df


def flag_pattern(df, direction):
    # Heuristic flag detection, not a confirmed chart pattern.
    if len(df) < 25:
        return False

    impulse = df.iloc[-18:-6]
    consolidation = df.iloc[-6:-1]

    if direction == "LONG":
        move = (
            impulse["close"].iloc[-1] /
            impulse["close"].iloc[0] - 1
        )
        high = impulse["high"].max()
        low = consolidation["low"].min()
        range_pct = (
            consolidation["high"].max() -
            consolidation["low"].min()
        ) / max(consolidation["close"].mean(), 1e-9)

        return move > 0.035 and range_pct < 0.035 and (
            consolidation["close"].iloc[-1] >
            consolidation["close"].iloc[0] * 0.97
        )

    move = (
        impulse["close"].iloc[-1] /
        impulse["close"].iloc[0] - 1
    )
    range_pct = (
        consolidation["high"].max() -
        consolidation["low"].min()
    ) / max(consolidation["close"].mean(), 1e-9)

    return move < -0.035 and range_pct < 0.035 and (
        consolidation["close"].iloc[-1] <
        consolidation["close"].iloc[0] * 1.03
    )


def analyze(symbol):
    d15 = candles(symbol, "15m", 220)
    d1h = candles(symbol, "1h", 220)
    d4h = candles(symbol, "4h", 220)

    if any(x is None or len(x) < 200 for x in [d15, d1h, d4h]):
        return None

    d15 = indicators(d15)
    d1h = indicators(d1h)
    d4h = indicators(d4h)

    # Use last fully closed candle.
    a = d15.iloc[-2]
    b = d1h.iloc[-2]
    c = d4h.iloc[-2]

    long_score = 0
    short_score = 0

    if a.ema20 > a.ema50:
        long_score += 15
    if a.ema20 < a.ema50:
        short_score += 15

    if a.close > a.ema200:
        long_score += 10
    if a.close < a.ema200:
        short_score += 10

    if b.ema20 > b.ema50:
        long_score += 15
    if b.ema20 < b.ema50:
        short_score += 15

    if c.ema20 > c.ema50:
        long_score += 15
    if c.ema20 < c.ema50:
        short_score += 15

    if a.macd > a.signal:
        long_score += 10
    if a.macd < a.signal:
        short_score += 10

    if 50 <= a.rsi <= 68:
        long_score += 10
    if 32 <= a.rsi <= 50:
        short_score += 10

    if a.vol_x >= 1.5:
        if a.close > a.open:
            long_score += 10
        elif a.close < a.open:
            short_score += 10

    long_flag = flag_pattern(d15, "LONG")
    short_flag = flag_pattern(d15, "SHORT")

    if long_flag:
        long_score += 15
    if short_flag:
        short_score += 15

    direction = "LONG" if long_score >= short_score else "SHORT"
    score = max(long_score, short_score)

    if score < MIN_SCORE:
        return None

    if direction == "LONG":
        entry = float(a.close)
        sl = float(d15["low"].iloc[-8:-1].min())
        risk = entry - sl
        if risk <= 0 or risk / entry > 0.04:
            return None
        tp1 = entry + risk * 1.5
        tp2 = entry + risk * 2.5
        pattern = "Bull Flag (heuristik)" if long_flag else "Trend/Momentum"
    else:
        entry = float(a.close)
        sl = float(d15["high"].iloc[-8:-1].max())
        risk = sl - entry
        if risk <= 0 or risk / entry > 0.04:
            return None
        tp1 = entry - risk * 1.5
        tp2 = entry - risk * 2.5
        pattern = "Bear Flag (heuristik)" if short_flag else "Trend/Momentum"

    return {
        "symbol": symbol,
        "direction": direction,
        "price": entry,
        "rsi": float(a.rsi),
        "volume": float(a.vol_x),
        "score": score,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "pattern": pattern
    }


def scan():
    exchange = api("/fapi/v1/exchangeInfo")
    tickers = api("/fapi/v1/ticker/24hr")

    if not exchange or not tickers:
        print("Gagal mengambil data Binance.")
        return []

    tradable = {
        x["symbol"] for x in exchange["symbols"]
        if x["quoteAsset"] == "USDT"
        and x["contractType"] == "PERPETUAL"
        and x["status"] == "TRADING"
    }

    active = [
        x for x in tickers
        if x["symbol"] in tradable
        and float(x.get("quoteVolume", 0)) > 0
    ]

    active.sort(
        key=lambda x: float(x["quoteVolume"]),
        reverse=True
    )

    symbols = [x["symbol"] for x in active[:TOP_COINS]]

    print("Memindai", len(symbols), "pair Futures...")
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}")
        try:
            result = analyze(symbol)
            if result:
                results.append(result)
        except Exception as e:
            print("Analisis gagal:", symbol, str(e)[:100])

        time.sleep(0.15)

    return sorted(results, key=lambda x: x["score"], reverse=True)


def format_message(results):
    now = datetime.now(timezone.utc).strftime("%d-%m-%Y %H:%M UTC")

    lines = [
        "🚀 BINANCE FUTURES SCANNER V7",
        "⏱ Timeframe: 15M | 1H | 4H",
        "🕒 " + now,
        "",
        "⚠️ Kandidat analisis, bukan sinyal entry pasti."
    ]

    for direction in ["LONG", "SHORT"]:
        selected = [
            x for x in results if x["direction"] == direction
        ][:10]

        lines += ["", "🟢 KANDIDAT " + direction]

        if not selected:
            lines.append("Belum ada kandidat.")
            continue

        for x in selected:
            lines += [
                "",
                f"{x['symbol']} | Score {x['score']}/100",
                f"Harga: {x['price']:.8g}",
                f"RSI: {x['rsi']:.1f} | Volume: {x['volume']:.2f}x",
                f"Pattern: {x['pattern']}",
                f"SL: {x['sl']:.8g}",
                f"TP1: {x['tp1']:.8g}",
                f"TP2: {x['tp2']:.8g}"
            ]

    lines += [
        "",
        "Gunakan konfirmasi candle dan cek spread sebelum entry.",
        "Leverage 10x meningkatkan risiko likuidasi."
    ]

    return "\n".join(lines)


def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram belum dikonfigurasi. Hasil scanner:")
        print(message)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    # Split messages to stay below Telegram's message limit.
    chunks = [message[i:i+3500] for i in range(0, len(message), 3500)]

    for chunk in chunks:
        response = requests.post(
            url,
            data={"chat_id": chat_id, "text": chunk},
            timeout=15
        )
        response.raise_for_status()


if __name__ == "__main__":
    print("Memulai Binance Futures Scanner V7...")
    results = scan()
    message = format_message(results)
    print(message)
    send_telegram(message)
    print("Scanner selesai.")
