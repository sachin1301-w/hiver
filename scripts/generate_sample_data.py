"""
Generates data/sample/sample_tweets.csv: a SYNTHETIC dataset that mirrors the
schema of the real Kaggle "Customer Support on Twitter" file (twcs.csv), for
one fictionalized brand handle, so the pipeline is runnable without the real
3M-row download.

This is NOT real data and must not be used as your golden evaluation set or
cited as the Kaggle dataset. Run data_prep.py against the real twcs.csv for
your actual submission.

Schema mirrors twcs.csv:
  tweet_id, author_id, inbound, created_at, text,
  response_tweet_id, in_response_to_tweet_id
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

BRAND = "GadgetCoSupport"

CUSTOMER_TEMPLATES = {
    "order_delivery_issue": [
        "Hey @{brand} my order #{oid} still hasn't arrived, it's been {days} days now",
        "@{brand} tracking says delivered but I never got my package. Order #{oid}",
        "package for order #{oid} arrived completely smashed, box was crushed",
        "@{brand} where is my order?? placed it {days} days ago, nothing",
    ],
    "refund_or_return": [
        "@{brand} I want to return order #{oid}, how do I start that",
        "still waiting on my refund for order #{oid}, it's been {days} days",
        "@{brand} can I exchange this for a different size? order #{oid}",
        "requested a refund last week for #{oid} and heard nothing back",
    ],
    "product_defect_quality": [
        "@{brand} the thing I bought stopped working after like 2 days, order #{oid}",
        "got the wrong item entirely, ordered X got Y. order #{oid}",
        "@{brand} this is defective right out of the box, order #{oid}",
        "screen is cracked and I haven't even dropped it, order #{oid}",
    ],
    "account_login_access": [
        "@{brand} I can't log into my account, keeps saying wrong password",
        "I think someone got into my account, can you help @{brand}",
        "@{brand} locked out of my account after too many tries, need help asap",
        "two factor code never arrives, can't get into my account",
    ],
    "billing_charge_issue": [
        "@{brand} I was charged twice for order #{oid}, please fix this",
        "there's a charge on my card I don't recognize from you guys",
        "@{brand} billed me for something I never ordered",
        "why was I charged {amount} more than the listed price on #{oid}",
    ],
    "general_inquiry_other": [
        "@{brand} do you guys ship to Canada?",
        "what are your store hours on weekends @{brand}",
        "@{brand} does the warranty cover water damage?",
        "is there a student discount available",
    ],
    "compliment_positive": [
        "just wanted to say thanks @{brand}, support fixed my issue super fast!",
        "@{brand} love the new product, best purchase this year",
        "shoutout to @{brand} support team, really appreciated the help today",
        "@{brand} you guys are great, quick and friendly service",
    ],
}

BRAND_REPLIES = {
    "order_delivery_issue": [
        "Hi there, sorry for the trouble! Please DM us your order number and zip code so we can look into the tracking. ^AK",
        "That's not the experience we want for you. Could you send us a DM with your order # so we can escalate to our shipping team? ^JM",
    ],
    "refund_or_return": [
        "We can definitely help with that. Please DM your order number and we'll get the return process started. ^AK",
        "Sorry for the delay on your refund! DM us your order # and we'll check the status right away. ^JM",
    ],
    "product_defect_quality": [
        "So sorry to hear that! Please DM us photos of the issue and your order number so we can send a replacement. ^AK",
        "That shouldn't happen — DM us your order number and we'll get this sorted with a replacement or refund. ^JM",
    ],
    "account_login_access": [
        "For account security we can't help over Twitter — please DM us or contact support@gadgetco.example directly so we can verify your identity. ^AK",
    ],
    "billing_charge_issue": [
        "We take billing issues seriously. Please DM your order number and we'll have our billing team review this right away. ^JM",
    ],
    "general_inquiry_other": [
        "Great question! Yes, we ship to Canada, delivery usually takes 5-7 business days. Let us know if you need anything else! ^AK",
        "We're open 9am-9pm every day including weekends! ^JM",
    ],
    "compliment_positive": [
        "This made our day! Thank you for the kind words 💙 ^AK",
        "So glad we could help! Thanks for being a customer 🙌 ^JM",
    ],
}


def rand_date():
    base = datetime(2023, 1, 1)
    return (base + timedelta(days=random.randint(0, 300), seconds=random.randint(0, 86400))).strftime(
        "%a %b %d %H:%M:%S +0000 %Y"
    )


def main(out_path: str, n_threads: int = 300):
    rows = []
    tweet_id = 1
    author_pool = [f"cust{i}" for i in range(1, 200)]

    intents = list(CUSTOMER_TEMPLATES.keys())
    weights = [3, 2, 2, 1, 1, 2, 1]  # roughly mimic real-world class imbalance

    for _ in range(n_threads):
        intent = random.choices(intents, weights=weights, k=1)[0]
        template = random.choice(CUSTOMER_TEMPLATES[intent])
        text = template.format(
            brand=BRAND,
            oid=random.randint(100000, 999999),
            days=random.randint(2, 21),
            amount=f"${random.randint(5, 80)}",
        )
        cust_id = random.choice(author_pool)
        cust_tweet_id = tweet_id
        tweet_id += 1

        brand_tweet_id = tweet_id
        tweet_id += 1
        reply_text = random.choice(BRAND_REPLIES[intent])

        rows.append(
            {
                "tweet_id": cust_tweet_id,
                "author_id": cust_id,
                "inbound": "True",
                "created_at": rand_date(),
                "text": text,
                "response_tweet_id": str(brand_tweet_id),
                "in_response_to_tweet_id": "",
            }
        )
        rows.append(
            {
                "tweet_id": brand_tweet_id,
                "author_id": BRAND,
                "inbound": "False",
                "created_at": rand_date(),
                "text": reply_text,
                "response_tweet_id": "",
                "in_response_to_tweet_id": str(cust_tweet_id),
            }
        )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "response_tweet_id",
                "in_response_to_tweet_id",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows ({n_threads} threads) to {out_path}")


if __name__ == "__main__":
    main("data/sample/sample_tweets.csv", n_threads=300)
