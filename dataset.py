"""Seeded synthetic orders. Run `python dataset.py` to reproduce and validate."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

SEED = 2026
RECORD_COUNT = 60
CATEGORIES = ("Apparel", "Electronics", "Home", "Footwear", "Beauty")
CATEGORY_WEIGHTS = (20, 15, 15, 20, 30)
STATUSES = ("Placed", "Shipped", "Delivered", "Returned", "Refunded")
STATUS_WEIGHTS = (15, 25, 40, 10, 10)
AMOUNT_RANGE_INR = (299, 14999)
DELAY_PROBABILITY = 0.20


def generate_orders(seed: int = SEED, count: int = RECORD_COUNT) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "record_id": f"NYK-{index:04d}",
            "category": rng.choices(CATEGORIES, weights=CATEGORY_WEIGHTS, k=1)[0],
            "status": rng.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0],
            "order_value_inr": rng.randint(*AMOUNT_RANGE_INR),
            "days_since_created": rng.randint(0, 30),
            # Historical shipment-delay flag; it can remain true after delivery/refund.
            "delayed_shipment": rng.random() < DELAY_PROBABILITY,
        }
        for index in range(1, count + 1)
    ]


def validate_orders(orders: list[dict]) -> dict:
    if len(orders) < 40:
        raise ValueError("At least 40 orders are required")
    expected = {
        "record_id",
        "category",
        "status",
        "order_value_inr",
        "days_since_created",
        "delayed_shipment",
    }
    ids = set()
    for row in orders:
        if set(row) != expected or row["record_id"] in ids:
            raise ValueError("Invalid schema or duplicate record ID")
        ids.add(row["record_id"])
        if row["category"] not in CATEGORIES or row["status"] not in STATUSES:
            raise ValueError("Unknown category or status")
        if type(row["days_since_created"]) is not int or not 0 <= row["days_since_created"] <= 30:
            raise ValueError("Age must be an integer from 0 to 30")
        if type(row["delayed_shipment"]) is not bool:
            raise ValueError("Delay flag must be boolean")
        if (
            type(row["order_value_inr"]) is not int
            or not AMOUNT_RANGE_INR[0] <= row["order_value_inr"] <= AMOUNT_RANGE_INR[1]
        ):
            raise ValueError("Order value is outside the designed range")
    categories = dict(sorted(Counter(row["category"] for row in orders).items()))
    statuses = dict(sorted(Counter(row["status"] for row in orders).items()))
    delayed_count = sum(row["delayed_shipment"] for row in orders)
    delayed_pct = 100 * delayed_count / len(orders)
    if any(categories.get(value, 0) < 3 for value in CATEGORIES):
        raise ValueError("Each category needs at least 3 records; select another seed")
    if any(statuses.get(value, 0) < 1 for value in STATUSES):
        raise ValueError("Every status must appear")
    if not 10 <= delayed_pct <= 30:
        raise ValueError("Delay distribution outside 10–30%; select another seed")
    ages = sorted(row["days_since_created"] for row in orders)
    import math

    return {
        "seed": SEED,
        "count": len(orders),
        "category_weights": dict(zip(CATEGORIES, CATEGORY_WEIGHTS, strict=True)),
        "status_weights": dict(zip(STATUSES, STATUS_WEIGHTS, strict=True)),
        "amount_range_inr": list(AMOUNT_RANGE_INR),
        "delay_probability": DELAY_PROBABILITY,
        "counts_per_category": categories,
        "counts_per_status": statuses,
        "delayed_count": delayed_count,
        "delayed_percentage": round(delayed_pct, 4),
        "age_p80_nearest_rank": ages[math.ceil(0.8 * len(ages)) - 1],
    }


ORDERS = generate_orders()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_orders(ORDERS), indent=2))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(ORDERS, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
